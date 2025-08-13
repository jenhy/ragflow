from . import prompts  # 从当前目录（prompts/）导入了名为 prompts.py 的模块。

# dir(prompts) 会返回 prompts.py 模块中所有公共属性（函数、变量、类等）的名称。
# 过滤掉了所有以 _ 开头的私有属性，确保只有公共接口被导出。
"""
如：
['ANALYZE_TASK_SYSTEM', 'ANALYZE_TASK_USER', 'CITATION_PLUS_TEMPLATE', 'CITATION_PROMPT_TEMPLATE', 'COMPLETE_TASK', 'CONTENT_TAGGING_PROMPT_TEMPLATE', 'CROSS_LANGUAGES_SYS_PROMPT_TEMPLATE', 'CROSS_LANGUAGES_USER_PROMPT_TEMPLATE', 'FULL_QUESTION_PROMPT_TEMPLATE', 'KEYWORD_PROMPT_TEMPLATE', 'NEXT_STEP', 'PROMPT_JINJA_ENV', 'QUESTION_PROMPT_TEMPLATE', 'RANK_MEMORY', 'REFLECT', 'STOP_TOKEN', 'SUMMARY4MEMORY', 'TAG_FLD', 'Tuple', 'VISION_LLM_DESCRIBE_PROMPT', 'VISION_LLM_FIGURE_DESCRIBE_PROMPT', 'analyze_task', 'chunks_format', 'citation_plus', 'citation_prompt', 'content_tagging', 'cross_languages', 'datetime', 'deepcopy', 'encoder', 'form_history', 'form_message', 'full_question', 'get_value', 'hash_str2int', 'jinja2', 'json', 'json_repair', 'kb_prompt', 'keyword_extraction', 'load_prompt', 'logging', 'message_fit_in', 'next_step', 'num_tokens_from_string', 'question_proposal', 'rank_memories', 're', 'reflect', 'tool_call_summary', 'tool_schema', 'vision_llm_describe_prompt', 'vision_llm_figure_describe_prompt']
"""
__all__ = [name for name in dir(prompts) if not name.startswith("_")]
# print(__all__)

# 是将 prompts.py 模块中的所有公共属性（函数、变量等）直接导入到 rag.prompts 这个包的全局命名空间中。
"""第1步：
   字典推导式：{name: getattr(prompts, name) for name in __all__}，会遍历 __all__ 列表中的每一个元素，并为每个元素创建一个键值对。
   name是键，值是getattr(prompts, name)。
   getattr() 是一个 Python 内置函数，它接受两个参数：一个对象和一个字符串。它会返回该对象中名为该字符串的属性。
   例如：getattr(prompts, 'ANALYZE_TASK_SYSTEM') 会返回 prompts.py 模块中名为 ANALYZE_TASK_SYSTEM 的属性（例如，一个字符串常量）。
   最终生成的字典大概是这样：
   {
    'ANALYZE_TASK_SYSTEM': getattr(prompts, 'ANALYZE_TASK_SYSTEM'),
    'ANALYZE_TASK_USER': getattr(prompts, 'ANALYZE_TASK_USER'),
    'CITATION_PLUS_TEMPLATE': getattr(prompts, 'CITATION_PLUS_TEMPLATE'),
    'CITATION_PROMPT_TEMPLATE': getattr(prompts, 'CITATION_PROMPT_TEMPLATE'),
    'COMPLETE_TASK': getattr(prompts, 'COMPLETE_TASK'),
    'CONTENT_TAGGING_PROMPT_TEMPLATE': getattr(prompts, 'CONTENT_TAGGING_PROMPT_TEMPLATE'),
    ...
    'question_proposal': getattr(prompts, 'question_proposal'), # 这是个函数
    'rank_memories': getattr(prompts, 'rank_memories'),       # 这是个函数
    ...
    're': getattr(prompts, 're'),                              # 这是一个导入的模块
    'json': getattr(prompts, 'json'),                            # 这是一个导入的模块
    ...
    }
    第2步：
    将这些函数或模块添加到全局命名空间中。
    globals() 是一个 Python 内置函数，它返回一个字典，代表了当前模块（在这里是 rag/prompts/__init__.py）的全局命名空间。
    这个字典包含了所有在当前作用域中定义的变量、函数和类。
    update() 是字典的一个方法，它接受另一个字典(第1步的结果)作为参数，并将该字典中的所有键值对添加到当前字典中。
    第1步和第2步结合起来的结果就是：
    把第1步生成的巨大字典中的所有键值对，全部添加到 rag/prompts/__init__.py 模块的全局命名空间中。
    在执行 globals().update(...) 之后，rag/prompts/__init__.py 模块的命名空间会变成：
        __all__ 变量被定义
        prompts 模块被导入
        QUESTION_PROMPT_TEMPLATE 这个变量被创建，它的值就是 prompts.QUESTION_PROMPT_TEMPLATE
        form_history 这个函数被创建，它的值就是 prompts.form_history
        ...以及 __all__ 列表中的所有其他名称。
    当从其他地方导入rag.prompts模块时，可以直接访问prompts模块中的所有名称，而不需要使用prompts.前缀。
    from rag import prompts
    
    # 无需再写 prompts.prompts.QUESTION_PROMPT_TEMPLATE
    my_prompt = prompts.QUESTION_PROMPT_TEMPLATE

    # 无需再写 prompts.prompts.form_history()
    history = prompts.form_history(...)
   """
globals().update({name: getattr(prompts, name) for name in __all__})
