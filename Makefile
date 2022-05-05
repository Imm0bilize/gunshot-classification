run_docker:
	@echo "Run docker"

install_req:
	@echo "Install requirements"
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

clean:
	echo ""

run:
	python -m src.main
