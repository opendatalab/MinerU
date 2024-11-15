import socket
from api import create_app
from pathlib import Path
import yaml


def get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(('8.8.8.8', 80))  # Google DNS 服务器
    ip_address = sock.getsockname()[0]
    sock.close()
    return ip_address


current_file_path = Path(__file__).resolve()
base_dir = current_file_path.parent
config_path = base_dir / "config/config.yaml"


class ConfigMap(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


with open(str(config_path), mode='r', encoding='utf-8') as fd:
    data = yaml.load(fd, Loader=yaml.FullLoader)
    _config = data.get(data.get("CurrentConfig", "DevelopmentConfig"))
config = ConfigMap()
for k, v in _config.items():
    config[k] = v
config['base_dir'] = base_dir
database = _config.get("database")
if database:
    if database.get("type") == "sqlite":
        database_uri = f'sqlite:///{base_dir}/{database.get("path")}'
    elif database.get("type") == "mysql":
        database_uri = f'mysql+pymysql://{database.get("user")}:{database.get("password")}@{database.get("host")}:{database.get("port")}/{database.get("database")}?'
    else:
        database_uri = ''
    config['SQLALCHEMY_DATABASE_URI'] = database_uri

ip_address = get_local_ip()
port = config.get("PORT", 5559)
# 配置 SERVER_NAME
config['SERVER_NAME'] = f'{ip_address}:{port}'
# 配置 APPLICATION_ROOT
config['APPLICATION_ROOT'] = '/'
# 配置 PREFERRED_URL_SCHEME
config['PREFERRED_URL_SCHEME'] = 'http'

app = create_app(config)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port, debug=config.get("DEBUG", False))
