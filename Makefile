build-api-server:
	docker build -f api_server.dockerfile -t mineru-api-server .
start-api-server:
	docker run -it --name mineru-api-server \
        -p 8000:8000 \
        mineru-api-server:latest bash
health:
	python3 api_server/api_manager.py health
list:
	python3 api_server/api_manager.py list
batch-dir:
	python3 api_server/api_manager.py batch-dir --input-dir /workspace/extracted_files/ --chunk-id 0001