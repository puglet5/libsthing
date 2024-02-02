### Packaging

#### Linux

Install Python 3.11 and patchelf then:
```bash
pip install pipenv

pipenv shell
pipenv install -d

python -m nuitka ./src/main.py --onefile --lto=yes 
```
