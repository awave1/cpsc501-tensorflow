# CPSC 501 - Tensorflow

The project is setup using [Pipenv](https://github.com/pypa/pipenv), to make it easier working with dependencies.

## Setup

### Using `pipenv`

Ensure you have `pipenv` installed, please see the [pipenv readme](https://github.com/pypa/pipenv/#installation).

If `pipenv` is installed, run the following in the project root,

```sh
pipenv install
```

That will take care of installing project dependencies as well as setting up a `virtualenv` for the project.

### Using `venv`

In the project root, run the following

```sh
python3 -m venv env

# or, if python3 is default
python -m venv env
```

Activate your virtual enviironment:

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

## Running the code

### `pipenv` scripts

To run a specific part,

```sh
# replace p1 with p2 or p3 to run that part of the assignment
pipenv run p1
```

To run the `predict` tests:

```sh
pipenv run predict:test
```

Please see the `Pipenv` scripts for other options.

### Without `pipenv`

To run a specific part,

```sh
python ./p1/MNISTStarter.py
# or
python ./p2/notMNISTStarter.py
```

To run the `predict` tests:

```sh
python ./predict_test.py
```
