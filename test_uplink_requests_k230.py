"""K230 上行速度测试（requests 库版）。

对照 Socket/https_client2.py 的做法，直接用 requests/urequests 库发请求，
不复用手写 socket+TLS 路径，看库内部路径是否也逐请求变慢。

部署：拷到 K230，改 CONFIG 后 exec(open(path).read()) 或 import。
"""
import time
import os
import gc

try:
    import requests
except ImportError:
    try:
        import urequests as requests
    except ImportError:
        requests = None

try:
    import urandom
except ImportError:
    try:
        import random as urandom
    except ImportError:
        urandom = None

try:
    import ujson as json
except ImportError:
    import json

try:
    import network
except ImportError:
    network = None

try:
    import usocket as socket_mod
except ImportError:
    import socket as socket_mod

try:
    import ussl as ssl_mod
except ImportError:
    try:
        import ssl as ssl_mod
    except ImportError:
        ssl_mod = None

if not hasattr(time, 'ticks_ms'):
    time.ticks_ms = lambda: int(time.time() * 1000)


# ============================================================================
# 证书验证开关
# ============================================================================
VERIFY_MODE = 'NONE'   # 'NONE'=关(跳证书验证)  'REQUIRED'=开(验链+主机名)

def _ssl_module():
    try:
        import ussl as m
        return m
    except ImportError:
        import ssl as m
        return m


def disable_cert_verify():
    """monkeypatch ussl/ssl：任何 SSLContext / wrap_socket 都强制 CERT_NONE。

    覆盖 requests 库内部不管用什么方式建 TLS 上下文。
    注意：K230 内置 ussl 模块可能不允许 setattr，失败时优雅跳过。
    """
    m = _ssl_module()
    print('[SSL ] 强制证书验证关闭 (CERT_NONE)')
    try:
        if hasattr(m, 'SSLContext') and hasattr(m, 'CERT_NONE'):
            _OrigCtx = m.SSLContext
            def _ctx(protocol):
                ctx = _OrigCtx(protocol)
                try:
                    ctx.verify_mode = m.CERT_NONE
                except Exception:
                    pass
                return ctx
            m.SSLContext = _ctx
            print('[SSL ] SSLContext 已 patch -> CERT_NONE')
    except Exception as e:
        print('[SSL ] SSLContext patch 失败(跳过): {}'.format(e))
    try:
        if hasattr(m, 'wrap_socket') and hasattr(m, 'CERT_NONE'):
            _orig_wrap = m.wrap_socket
            def _wrap(sock, *a, **kw):
                kw['cert_reqs'] = m.CERT_NONE
                kw.pop('ca_certs', None)
                try:
                    kw['check_hostname'] = False
                except Exception:
                    pass
                return _orig_wrap(sock, *a, **kw)
            m.wrap_socket = _wrap
            print('[SSL ] wrap_socket 已 patch -> CERT_NONE')
    except Exception as e:
        print('[SSL ] wrap_socket patch 失败(跳过): {}'.format(e))


# ============================================================================
# 配置区
# ============================================================================
TARGET_BASE   = '/Mtpi'
TARGET_BIZ    = '/mouseVideoUpload'
TARGET_URL    = 'https://otapi.kukac.cloud'
TARGET_DEV_SN = 'K230_REQTEST_001'
TARGET_CHUNK  = 262144              # 256KB

CONTROL_URL = 'https://www.baidu.com'

TEST_SIZE    = 262144              # 单分片 256KB
TEST_ROUNDS  = 3

TEMP_FILE   = '/sdcard/_reqspeed.bin'

KEEPALIVE_RADIO = True

# True = 复用同一条 TCP+TLS 连接发 init/chunk/complete（长连接）
# False = 每个请求新建连接（短连接）
USE_KEEPALIVE = True


# ============================================================================
# 网络激活
# ============================================================================
def ensure_network(timeout_sec=20):
    if network is None:
        print('[NET] PC 调试模式')
        return True
    try:
        lan = network.LAN(3)
        if not lan.active():
            lan.active(True)
            time.sleep(2)
        cfg = lan.ifconfig()
        ip, dns = cfg[0], cfg[3]
        if (not ip or ip == '0.0.0.0') or (not dns or dns == '0.0.0.0'):
            lan.ifconfig('dhcp')
            t0 = time.time()
            while time.time() - t0 < timeout_sec:
                cfg = lan.ifconfig()
                if cfg[0] != '0.0.0.0' and cfg[3] != '0.0.0.0':
                    break
                time.sleep(0.5)
        print('[NET] ip={} dns={}'.format(cfg[0], cfg[3]))
    except Exception as e:
        print('[NET] LAN 异常: {}'.format(e))
        return False
    try:
        import net_manager
        net_manager.net_bringup('reqtest')
    except Exception:
        pass
    return True


# ============================================================================
# 4G 射频保活
# ============================================================================
_ka_addr = None
_last_net_ms = 0

def _radio_keepalive():
    global _ka_addr, _last_net_ms
    now = time.ticks_ms()
    if now - _last_net_ms < 1500:
        return
    try:
        import usocket as socket_mod
    except ImportError:
        import socket as socket_mod
    if _ka_addr is None:
        try:
            _ka_addr = socket_mod.getaddrinfo('otapi.kukac.cloud', 80)[0][-1]
        except Exception:
            return
    try:
        s = socket_mod.socket()
        s.settimeout(2)
        s.connect(_ka_addr)
        s.close()
    except Exception:
        pass
    _last_net_ms = time.ticks_ms()


# ============================================================================
# RNG 探针：验证 mbedTLS 熵源是否逐次变慢
# ============================================================================
def rng_probe():
    """测一次 os.urandom(32) 耗时。若随请求次数显著增大 → 熵源枯竭。"""
    if urandom is None:
        print('  [RNG  ] 无 urandom 模块')
        return -1
    t0 = time.ticks_ms()
    try:
        os.urandom(32)
    except Exception as e:
        print('  [RNG  ] FAIL {}'.format(e))
        return -1
    ms = time.ticks_ms() - t0
    print('  [RNG  ] urandom(32)={}ms'.format(ms))
    return ms


# ============================================================================
# requests 封装：按请求计时，兼容/不兼容 verify 参数
# ============================================================================
def _call(method, url, headers, body):
    """调用 requests，返回 (status, resp_json_or_none, total_ms)。

    先尝试 verify=False（跳证书验证），库不支持该参数则回退不带。
    """
    fn = getattr(requests, method)
    kw = {'headers': headers}
    if body is not None:
        kw['data'] = body
    rng_probe()
    t0 = time.ticks_ms()
    resp = fn(url, **kw)
    total = time.ticks_ms() - t0
    status = getattr(resp, 'status_code', 0)
    parsed = None
    try:
        parsed = resp.json()
    except Exception:
        try:
            txt = resp.text
            if txt:
                parsed = json.loads(txt)
        except Exception:
            parsed = None
    resp.close()
    return status, parsed, total


def _biz_ok(parsed):
    return isinstance(parsed, dict) and parsed.get('code') == 200 \
        and parsed.get('status') is True


# ============================================================================
# 长连接会话：一条 TCP+TLS 复用，HTTP/1.1 keep-alive
# ============================================================================
class KeepAliveSession:
    """复用同一 socket 发多个 HTTP/1.1 请求。

    request() 每次从写请求开始计时到读完响应，返回 (status, body_bytes, total_ms)。
    """

    def __init__(self, host, port, timeout=30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.ssl = None

    def open(self):
        if self.sock is not None:
            return True
        if ssl_mod is None:
            print('  [KEEP] 无 SSL 模块，无法长连接')
            return False
        try:
            ai = socket_mod.getaddrinfo(self.host, self.port)[0][-1]
            s = socket_mod.socket()
            s.settimeout(self.timeout)
            if s.connect(ai) is False:
                raise OSError('connect False')
            if hasattr(ssl_mod, 'SSLContext'):
                proto = getattr(ssl_mod, 'PROTOCOL_TLS_CLIENT', None)
                if proto is None:
                    proto = getattr(ssl_mod, 'PROTOCOL_TLS', None)
                if proto is None:
                    raise OSError('无 SSLContext protocol 常量')
                ctx = ssl_mod.SSLContext(proto)
                try:
                    ctx.verify_mode = ssl_mod.CERT_NONE
                except Exception:
                    pass
                self.ssl = ctx.wrap_socket(s, server_hostname=self.host)
            else:
                self.ssl = ssl_mod.wrap_socket(s, server_hostname=self.host)
            self.sock = s
            return True
        except Exception as e:
            print('  [KEEP] open FAIL: {}'.format(e))
            self.close()
            return False

    def close(self):
        try:
            if self.ssl is not None:
                self.ssl.close()
        except Exception:
            pass
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.ssl = None
        self.sock = None

    def request(self, method, path, headers, body):
        """发一个 keep-alive 请求。返回 (status, resp_body, total_ms)。"""
        if not self.open():
            return 0, b'', 0
        t0 = time.ticks_ms()
        try:
            body_bytes = body.encode('utf-8') if isinstance(body, str) else (body or b'')
            hdr = ''
            has_clen = False
            for k, v in headers.items():
                hdr += '{}: {}\r\n'.format(k, v)
                if k.lower() == 'content-length':
                    has_clen = True
            if not has_clen:
                hdr += 'Content-Length: {}\r\n'.format(len(body_bytes))
            req = (
                '{} {} HTTP/1.1\r\n'
                'Host: {}\r\n'
                'Connection: keep-alive\r\n'
                '{}\r\n'
            ).format(method, path, self.host, hdr).encode('utf-8') + body_bytes

            # ---- write ----
            tw = time.ticks_ms()
            sent = 0
            while sent < len(req):
                n = self.ssl.write(req[sent:])
                if n is None:
                    n = 0
                if n <= 0:
                    time.sleep(0.01)
                    continue
                sent += n
            ms_write = time.ticks_ms() - tw

            # ---- read status + headers ----
            th = time.ticks_ms()
            buf = b''
            while b'\r\n\r\n' not in buf:
                chunk = self.ssl.read(1024)
                if not chunk:
                    raise OSError('conn closed during header')
                buf += chunk
            ms_header = time.ticks_ms() - th
            head, _, rest = buf.partition(b'\r\n\r\n')
            status = 0
            parts = head.split(b'\r\n', 1)[0].split(b' ')
            if len(parts) >= 2:
                try:
                    status = int(parts[1])
                except ValueError:
                    pass
            clen = -1
            chunked = False
            for line in head.split(b'\r\n'):
                l = line.lower()
                if l.startswith(b'content-length:'):
                    try:
                        clen = int(line.split(b':', 1)[1].strip())
                    except ValueError:
                        clen = -1
                elif l.startswith(b'transfer-encoding:'):
                    if b'chunked' in l:
                        chunked = True

            body_out = b''
            if chunked:
                body_out = self._read_chunked(rest)
            elif clen >= 0:
                body_out = rest
                while len(body_out) < clen:
                    chunk = self.ssl.read(1024)
                    if not chunk:
                        break
                    body_out += chunk
                body_out = body_out[:clen]
            else:
                body_out = rest
                while True:
                    chunk = self.ssl.read(1024)
                    if not chunk:
                        break
                    body_out += chunk
            total = time.ticks_ms() - t0
            print('  [KEEP] {} {}{}ms write={}ms header={}ms body={}ms clen={} chunked={}'.format(
                method, path,
                '' if status else 'FAIL',
                ms_write, ms_header, total - ms_write - ms_header, clen, chunked))
            return status, body_out, total
        except Exception as e:
            total = time.ticks_ms() - t0
            print('  [KEEP] request FAIL: {} ({}ms)'.format(e, total))
            self.close()
            return 0, b'', total

    def _read_chunked(self, initial=b''):
        """解析 HTTP chunked 响应体。initial 为头部之后已读到的字节。"""
        data = initial
        out = b''
        while True:
            # 取 chunk size 行（以 \r\n 结尾的十六进制）
            while b'\r\n' not in data:
                chunk = self.ssl.read(1024)
                if not chunk:
                    return out
                data += chunk
            size_line, _, data = data.partition(b'\r\n')
            try:
                size = int(size_line.split(b';')[0].strip(), 16)
            except ValueError:
                return out
            if size == 0:
                return out
            while len(data) < size + 2:
                chunk = self.ssl.read(1024)
                if not chunk:
                    return out
                data += chunk
            out += data[:size]
            data = data[size + 2:]


def _fmt_keep(k):
    return '{}ms'.format(k)


def _full_path(action):
    return TARGET_BASE + TARGET_BIZ + action


def _req_url(action):
    return TARGET_URL + _full_path(action)


def TARGET_HOST_FROM_URL():
    u = TARGET_URL.split('://', 1)[1]
    if '/' in u:
        u = u.split('/', 1)[0]
    if ':' in u:
        u = u.split(':')[0]
    return u


# ============================================================================
# 对照组：baidu GET（requests 库）
# ============================================================================
def control_get():
    if KEEPALIVE_RADIO:
        _radio_keepalive()
    rng_probe()
    t0 = time.ticks_ms()
    try:
        resp = requests.get(CONTROL_URL + '/', timeout=15)
        status = getattr(resp, 'status_code', 0)
        resp.close()
        total = time.ticks_ms() - t0
        print('  [CTRL ] GET {} status={} total={}ms'.format(CONTROL_URL, status, total))
        return total
    except Exception as e:
        total = time.ticks_ms() - t0
        print('  [CTRL ] GET {} FAIL {} total={}ms'.format(CONTROL_URL, e, total))
        return -1


# ============================================================================
# 分片上传协议（全部走 requests 库）
# ============================================================================
def do_init(filename, file_size, chunk_size):
    body = json.dumps({
        'uploadId': '', 'devSerial': TARGET_DEV_SN,
        'fileSize': float(file_size), 'filename': filename,
        'chunkSize': int(chunk_size),
    })
    status, parsed, total = _call(
        'post', _req_url('/init'),
        {'Content-Type': 'application/json'}, body.encode('utf-8'))
    print('  [INIT ] total={}ms status={}'.format(total, status))
    if status != 200 or not _biz_ok(parsed):
        print('  [INIT ] FAIL biz={!r}'.format(parsed))
        return None
    data = parsed.get('data') or {}
    uid = data.get('uploadId') or ''
    print('  [INIT ] ok uid={:.16} chunk={}B'.format(uid, data.get('chunkSize', '?')))
    return uid, int(data.get('offset', 0)), int(data.get('chunkSize', 0) or chunk_size)


def do_chunk(upload_id, chunk_data, offset, part_no):
    boundary = '----K230R' + str(time.ticks_ms())[-8:]
    head = (
        '--' + boundary + '\r\n'
        'Content-Disposition: form-data; name="chunk"; filename="blob"\r\n'
        'Content-Type: application/octet-stream\r\n\r\n'
    ).encode('utf-8')
    tail = ('\r\n--' + boundary + '--\r\n').encode('utf-8')
    mbody = head + bytes(chunk_data) + tail
    status, parsed, total = _call(
        'put', _req_url('/chunk'),
        {'Content-Type': 'multipart/form-data; boundary=' + boundary,
         'X-Offset': str(offset), 'X-Upload-Id': upload_id,
         'X-part-number': str(part_no)},
        mbody)
    data_kb = len(chunk_data) / 1024.0
    kbps = data_kb / (total / 1000.0) if total > 0 else 0
    print('  [CHUNK] part={} {}KB {}KB/s total={}ms status={}'.format(
        part_no, int(data_kb), kbps, total, status))
    if status != 200 or not _biz_ok(parsed):
        print('  [CHUNK] FAIL offset={} biz={!r}'.format(offset, parsed))
        return None
    data = parsed.get('data')
    new_off = data.get('offset') if isinstance(data, dict) else data
    return int(new_off) if new_off is not None else offset + len(chunk_data)


def do_complete(upload_id):
    body = 'uploadId={}'.format(upload_id)
    status, parsed, total = _call(
        'post', _req_url('/complete'),
        {'Content-Type': 'application/x-www-form-urlencoded'},
        body.encode('utf-8'))
    print('  [COMP ] total={}ms status={}'.format(total, status))
    return status == 200 and _biz_ok(parsed)


# ============================================================================
# 分片上传协议（长连接版：复用同一连接）
# ============================================================================
def ks_init(sess, filename, file_size, chunk_size):
    body = json.dumps({
        'uploadId': '', 'devSerial': TARGET_DEV_SN,
        'fileSize': float(file_size), 'filename': filename,
        'chunkSize': int(chunk_size),
    })
    status, resp, total = sess.request(
        'POST', _full_path('/init'),
        {'Content-Type': 'application/json'}, body)
    parsed = json.loads(resp.decode('utf-8')) if resp else None
    print('  [INIT ] total={}ms status={}'.format(total, status))
    if status != 200 or not _biz_ok(parsed):
        print('  [INIT ] FAIL biz={!r}'.format(parsed))
        return None
    data = parsed.get('data') or {}
    uid = data.get('uploadId') or ''
    print('  [INIT ] ok uid={:.16} chunk={}B'.format(uid, data.get('chunkSize', '?')))
    return uid, int(data.get('offset', 0)), int(data.get('chunkSize', 0) or chunk_size)


def ks_chunk(sess, upload_id, chunk_data, offset, part_no):
    boundary = '----K230K' + str(time.ticks_ms())[-8:]
    head = (
        '--' + boundary + '\r\n'
        'Content-Disposition: form-data; name="chunk"; filename="blob"\r\n'
        'Content-Type: application/octet-stream\r\n\r\n'
    ).encode('utf-8')
    tail = ('\r\n--' + boundary + '--\r\n').encode('utf-8')
    mbody = head + bytes(chunk_data) + tail
    status, resp, total = sess.request(
        'PUT', _full_path('/chunk'),
        {'Content-Type': 'multipart/form-data; boundary=' + boundary,
         'X-Offset': str(offset), 'X-Upload-Id': upload_id,
         'X-part-number': str(part_no)},
        mbody)
    parsed = json.loads(resp.decode('utf-8')) if resp else None
    data_kb = len(chunk_data) / 1024.0
    kbps = data_kb / (total / 1000.0) if total > 0 else 0
    print('  [CHUNK] part={} {}KB {}KB/s total={}ms status={}'.format(
        part_no, int(data_kb), kbps, total, status))
    if status != 200 or not _biz_ok(parsed):
        print('  [CHUNK] FAIL offset={} biz={!r}'.format(offset, parsed))
        return None
    data = parsed.get('data')
    new_off = data.get('offset') if isinstance(data, dict) else data
    return int(new_off) if new_off is not None else offset + len(chunk_data)


def ks_complete(sess, upload_id):
    body = 'uploadId={}'.format(upload_id)
    status, resp, total = sess.request(
        'POST', _full_path('/complete'),
        {'Content-Type': 'application/x-www-form-urlencoded'},
        body)
    print('  [COMP ] total={}ms status={}'.format(total, status))
    parsed = json.loads(resp.decode('utf-8')) if resp else None
    return status == 200 and _biz_ok(parsed)


# ============================================================================
# 主流程
# ============================================================================
def _make_temp_file(path, size):
    parent = path.rsplit('/', 1)[0]
    if parent.startswith('/'):
        try:
            os.stat(parent)
        except OSError:
            path = path.rsplit('/', 1)[-1]
    with open(path, 'wb') as f:
        remaining = size
        while remaining > 0:
            w = min(65536, remaining)
            f.write(bytes(w))
            remaining -= w
    return path


def run():
    if requests is None:
        print('[ABORT] 设备无 requests/urequests 库')
        return

    print('=' * 72)
    print('K230 上行测试(requests库)  target={}{}'.format(TARGET_URL, _full_path('')))
    print('对照: {}  射频保活={}'.format(CONTROL_URL, '开' if KEEPALIVE_RADIO else '关'))
    print('证书验证: {}  长连接: {}'.format(
        VERIFY_MODE, '开(复用连接)' if USE_KEEPALIVE else '关(每请求新建)'))
    print('=' * 72)

    # 打印 requests 库真实来源（确认是哪个实现）
    for attr in ('__file__', '__name__'):
        try:
            print('[REQ ] {}.{} = {}'.format('requests', attr, getattr(requests, attr)))
        except Exception:
            pass

    if VERIFY_MODE == 'NONE':
        disable_cert_verify()

    if not ensure_network():
        print('[ABORT] 网络未就绪')
        return

    size = TEST_SIZE
    size_kb = size / 1024.0
    n_chunks = (size + TARGET_CHUNK - 1) // TARGET_CHUNK
    print('')
    print('-' * 72)
    print('file={}KB ({} chunks x {}KB)  rounds={}'.format(
        int(size_kb), n_chunks, TARGET_CHUNK // 1024, TEST_ROUNDS))
    print('-' * 72)

    file_path = _make_temp_file(TEMP_FILE, size)
    filename = file_path.rsplit('/', 1)[-1]
    all_timings = []

    for rnd in range(TEST_ROUNDS):
        gc.collect()
        print('')
        print('[round {}/{}]'.format(rnd + 1, TEST_ROUNDS))

        control_get()

        t_round = time.ticks_ms()
        sess = None
        try:
            if USE_KEEPALIVE:
                sess = KeepAliveSession(TARGET_HOST_FROM_URL(), 443)
                if not sess.open():
                    print('  [ABORT] 长连接建立失败')
                    continue
                init_res = ks_init(sess, filename, size, TARGET_CHUNK)
                if not init_res:
                    continue
                uid, offset, eff_chunk = init_res

                gc.collect()
                with open(file_path, 'rb') as f:
                    part_no = 0
                    while offset < size:
                        chunk_data = f.read(eff_chunk)
                        if not chunk_data:
                            break
                        gc.collect()
                        new_off = ks_chunk(sess, uid, chunk_data, offset, part_no)
                        if new_off is None:
                            break
                        offset = new_off
                        part_no += 1
                    else:
                        gc.collect()
                        ok = ks_complete(sess, uid)
                        total_ms = time.ticks_ms() - t_round
                        kbps = size_kb / (total_ms / 1000.0)
                        print('  [DONE ] {} total={}ms -> {:.1f} KB/s'.format(
                            'ok' if ok else 'COMPLETE_FAIL', total_ms, kbps))
                        all_timings.append(total_ms)
                        continue
                print('  [ABORT] chunk 失败，跳过本轮')
            else:
                init_res = do_init(filename, size, TARGET_CHUNK)
                if not init_res:
                    continue
                uid, offset, eff_chunk = init_res

                gc.collect()
                with open(file_path, 'rb') as f:
                    part_no = 0
                    while offset < size:
                        chunk_data = f.read(eff_chunk)
                        if not chunk_data:
                            break
                        gc.collect()
                        new_off = do_chunk(uid, chunk_data, offset, part_no)
                        if new_off is None:
                            break
                        offset = new_off
                        part_no += 1
                    else:
                        gc.collect()
                        ok = do_complete(uid)
                        total_ms = time.ticks_ms() - t_round
                        kbps = size_kb / (total_ms / 1000.0)
                        print('  [DONE ] {} total={}ms -> {:.1f} KB/s'.format(
                            'ok' if ok else 'COMPLETE_FAIL', total_ms, kbps))
                        all_timings.append(total_ms)
                        continue
                print('  [ABORT] chunk 失败，跳过本轮')
        except Exception as e:
            print('  [ERR] {}: {}'.format(type(e).__name__, e))
        finally:
            if sess is not None:
                sess.close()

    try:
        os.remove(file_path)
    except Exception:
        pass

    if all_timings:
        avg = sum(all_timings) / len(all_timings)
        avg_kbps = size_kb / (avg / 1000.0)
        print('')
        print('  平均: {:.0f}ms -> {:.1f} KB/s'.format(avg, avg_kbps))

    print('')
    print('=' * 72)
    print('观察每行 total：requests 库内部路径是否也逐请求变慢')
    print('=' * 72)


if __name__ == '__main__':
    run()
