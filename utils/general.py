import yaml


class Config():
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            self._attr = yaml.load(f, Loader=yaml.FullLoader)["settings"]

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, item):
        try:
            return self._attr[item]
        except KeyError:
            try:
                return self.__dict__[item]
            except KeyError:
                return None

    def __str__(self):
        print("##########   CONFIGURATION INFO   ##########")
        pretty(self._attr)
        return '\n'


def pretty(d, indent=0):
    for key, value in d.items():
        print('    ' * indent + str(key) + ':', end='')
        if isinstance(value, dict):
            print()
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))