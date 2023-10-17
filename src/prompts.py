'''
===========================================
        Module: Prompts collection
===========================================
'''
# Note: Precise formatting of spacing and indentation of the prompt template is important for Llama-2-7B-Chat,
# as it is highly sensitive to whitespace changes. For example, it could have problems generating
# a summary from the pieces of context if the spacing is not done correctly

qa_template = """Verwende die folgenden Informationen, um die Frage des Benutzers zu beantworten.
Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt, versuche nicht, eine Antwort zu erfinden.

Kontext: {context}
Frage: {question}

Gebe nur die hilfreiche Antwort unten zurück und nichts anderes.
Hilfreiche Antwort:
"""
