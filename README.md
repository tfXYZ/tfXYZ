# tfXYZ

A framework on top of tensorflow.

## Dependencies

* Python 2.7/3.5+
* Python libraries: Tensorflow 1.0, scipy, sklearn, arff

## Usage

To simply execute the program, type

```
python main.py
```

Most of the behavior is determined by flags. To see a list of all flags do

```
python main.py --help
```

### Adding new "apps"

An *app* defines what the network is learning and how the result should be displayed. To add a new app add a python class that inherits from `BaseApp` in `apps/base.py`. The new class has to be in a python script in the subdirectory `apps`.

#### Example
Add a class called "Mnist" in `apps/digits.py`. Run it with

```
python main.py --apps="digits.Mnist"
```
