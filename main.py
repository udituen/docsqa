import streamlit as st




uploaded_file = st.file_uploader("Upload your file", type=["txt","pdf"])
if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)
    data = uploaded_file.read()

    if uploaded_file.type == "text/plain":
        st.text_area("Content", data.decode("utf-8"), height=300)
    else:
        st.info(f"Uploaded {len(data)} bytes (PDF or other format)")
