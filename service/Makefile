all:
	@echo "commands: format, lint, install_dependencies run"

format:
	autopep8 --in-place src/*.py --max-line-length 80

lint:
	pylint src/*.py --extension-pkg-whitelist='pydantic'

install_dependencies:
	sudo pip install -r requirements.txt

run:
	sudo docker-compose up -d --build
