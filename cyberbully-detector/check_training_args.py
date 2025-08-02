import transformers.training_args as ta
print(ta.__file__)
print(dir(ta))
print(hasattr(ta.TrainingArguments.__init__, '__code__))
print(ta.TrainingArguments.__init__.__code__.co_varnames)