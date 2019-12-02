Create virtual environment

```sh
python3 -m venv env
```

Activate your virtual enviironment

```sh
source env/bin/activate

# or
source env/bin/activate.fish   # (pick extension for your shell e.g. .fish, .csh)
```

Installing packages:

```sh
pip install -r requirements.txt
```

Ensure that `virtualenv` is setup properly:

```sh
which python
# You should see something like this: ./env/bin/python
```

To exit `virtualenv`:

```sh
deactivate
```
