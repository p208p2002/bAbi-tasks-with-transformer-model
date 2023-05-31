import re

def remove_punctuation_wo_comma(text):
    """移除標點符號，但是不包含逗號"""
    cleaned_text = re.sub(r"[^\w\s!,]", "", text)
    return cleaned_text


def parse_answer(text: str,eos_token):
    text = text.replace(eos_token,"")
    try:
        answer_start = text.index("Answer:")
        answer = remove_punctuation_wo_comma(text[answer_start:].split(":")[1].strip().split("\n")[0])
    except:
        answer = ""
    return answer