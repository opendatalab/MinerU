#!/bin/bash
###
 # @Author: FutureMeng futuremeng@gmail.com
 # @Date: 2025-02-14 09:05:04
 # @LastEditors: FutureMeng futuremeng@gmail.com
 # @LastEditTime: 2025-02-14 09:08:07
 # @FilePath: /MinerU/services/fastapi/app/start.sh
 # @Description: 
 # Copyright (c) 2025 by Jiulu.ltd, All Rights Reserved.
### 
echo "starting miner server"
source /opt/mineru_venv/bin/activate
cd /gateway
uvicorn app.main:app --host 0.0.0.0 --port 80