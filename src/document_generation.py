import streamlit as st
from ctransformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

def document_generation():
    #Define the Streamlit app layout
    st.title("Dokument Generator mit Llama 2")

    with st.form('Form zur Generierung eines Dokumentes'):
        # User input for letter of dismissal
        st.markdown("### Details")
        doctor_name = st.text_input("Name des behandelnen Arztes")
        patient_name = st.text_input("Name des Patienten")
        hospital = st.text_input("Name des Krankenhauses")
        history = st.text_area("Beschreiben sie den Aufenthaltsverlauf")

        #Generate LLM repsonse
        generate_cover_letter = st.form_submit_button("Generate Cover Letter")

        if generate_cover_letter:

            prompt_format = """Du bist ein hilfreicher Assistant. Du wandelst Eckdaten in ein fertiges Dokument um.
            Du gibst ausschließlich das fertige Dokument zurück und nichts anderes. Die Eckdaten lauten wie folgt:
            Der Aufenthaltsverlauf des Patienten: {history}\n
            Der Name des Arztes der im Anschreiben angegeben werden soll: {doctor_name}\n
            Der Name des Patienten, um den es geht: {patient_name}\n
            Das Krankenhaus, bei dem der Patient behandelt wurde: {hospital}\n
            Generiere daraus das Dokument:"""

            model = AutoModelForCausalLM.from_pretrained("TheBloke/leo-hessianai-7B-chat-GGUF", model_file="leo-hessianai-7b-chat.Q5_K_M.gguf", model_type="llama", hf=True)
            tokenizer = AutoTokenizer.from_pretrained(model)  
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

            with st.spinner("Generiere Dokument..."):
                response = pipe(prompt_format.format(history = history, doctor_name = doctor_name, patient_name = patient_name, hospital = hospital), do_sample=True, top_p=0.95, max_new_tokens=256)
                print(response)
                generated_cover_letter = " ".join([item for item in response['generated_text']])

            st.subheader("Generated Cover Letter:")
            st.write(generated_cover_letter)
            # Offering download link for generated cover letter
            st.subheader("Download Generated Cover Letter:")
            st.download_button("Download Cover Letter as TXT", generated_cover_letter, key="cover_letter")


