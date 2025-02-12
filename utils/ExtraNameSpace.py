import types
from typing import Callable


class NameSpace:
    __instance = None
    _args = None

    def __init__(self):
        if self.__instance is None:
            self.function_map = dict()
            NameSpace.__instance = self
        else:
            raise Exception("cannot instantiate a virtual Namespace again")

    @staticmethod
    def get_instance():
        if NameSpace.__instance is None:
            NameSpace()
        return NameSpace.__instance

    @classmethod
    def register(cls, fn_name: str):
        def decorator(fn):
            func = Function(fn=fn, space_cls=cls)
            name = func.register_key(fn_name=fn_name)  # 构造注册命名空间的name
            cls_instance = DatasetsReaderNameSpace.get_instance()
            cls_instance.function_map[name] = fn
            return func

        return decorator

    @classmethod
    def get_function(cls, fn: Callable, attr_name: str) -> Callable:  # 先屯着
        cls_instance = DatasetsReaderNameSpace.get_instance()
        func = Function(fn=fn, space_cls=cls)
        fn_name = getattr(NameSpace._args, attr_name, "Default")
        name = func.register_key(fn_name=fn_name)

        fn = cls_instance.function_map.get(name)

        if not fn:
            fn_name = "Default"  # 大不了就默认如果找不到就要求给Default。抛出异常了也没问题
            name = func.register_key(fn_name=fn_name)
            fn = cls_instance.function_map.get(name)

        return fn

    @classmethod
    def get(cls, fn: Callable) -> Callable:
        return cls.get_function(fn=fn, attr_name='dataset')


class Function(object):
    def __init__(self, fn: Callable, space_cls):
        self.fn: object = fn
        self.space_cls = space_cls

    def __get__(self, instance, owner):
        if instance is not None:
            return types.MethodType(self, instance)
        else:
            return self

    def __call__(self, *args, **kwargs):
        fn = self.space_cls.get(self.fn)
        if not fn:
            raise Exception("no matching function found.")
        # invoking the wrapped function and returning the value.
        return fn(*args, **kwargs)

    def register_key(self, fn_name=None):
        return tuple([
            # self.fn.__module__,
            self.fn.__class__,
            self.fn.__name__,  # 这个key目前无意义，但似乎不需要额外继承出DateReaderFunction之类的类
            self.space_cls.__name__,
            fn_name
        ])


class DatasetsReaderNameSpace(NameSpace):
    @classmethod
    def get(cls, fn: Callable) -> Callable:
        return cls.get_function(fn=fn, attr_name='dataset')


class PredictionCleanNameSpace(NameSpace):
    pass


class ScoreNameSpace(NameSpace):
    pass


class KnowledgeExtractionNameSpace(NameSpace):

    @classmethod
    def get(cls, fn: Callable) -> Callable:
        return cls.get_function(fn, 'cot_trigger_type')


class PromptMethodNameSpace(NameSpace):
    @classmethod
    def get(cls, fn: Callable) -> Callable:
        return cls.get_function(fn, 'train_prompt_type')
