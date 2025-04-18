import streamlit as st

def get_drive(credentials_file):
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    gauth = GoogleAuth()
    gauth.DEFAULT_SETTINGS['client_config_file'] = credentials_file
    gauth.LoadCredentialsFile(credentials_file)
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(credentials_file)
    drive = GoogleDrive(gauth)
    return drive

def collaborative_experiments_panel():
    st.sidebar.markdown('---')
    with st.sidebar.expander('Collaborative Experiments (Google Drive)', expanded=False):
        st.markdown('**Save or load experiments to Google Drive.**')
        credentials_file = None
        if 'drive_credentials' not in st.session_state:
            cred_file = st.file_uploader('Upload Google client_secrets.json', type='json', key='drive_creds')
            if cred_file:
                with open('client_secrets.json', 'wb') as f:
                    f.write(cred_file.read())
                st.session_state['drive_credentials'] = 'client_secrets.json'
        if 'drive_credentials' in st.session_state:
            credentials_file = st.session_state['drive_credentials']
            drive = get_drive(credentials_file)
            # Save experiment
            if st.button('Save Experiment to Drive'):
                import pickle
                import time
                exp_bytes = pickle.dumps(dict(st.session_state))
                fname = f"experiment_{int(time.time())}.pkl"
                file_drive = drive.CreateFile({'title': fname})
                file_drive.SetContentString(exp_bytes.hex())
                file_drive.Upload()
                st.success(f'Experiment uploaded! File ID: {file_drive["id"]}')
            # Load experiment
            file_id = st.text_input('Google Drive File ID to Load', '')
            if st.button('Load Experiment from Drive') and file_id:
                file_drive = drive.CreateFile({'id': file_id})
                file_drive.FetchMetadata(fetch_all=True)
                file_drive.GetContentFile('loaded_experiment.pkl')
                with open('loaded_experiment.pkl', 'rb') as f:
                    loaded_state = pickle.loads(bytes.fromhex(f.read().decode()))
                st.session_state.update(loaded_state)
                st.success('Experiment loaded from Google Drive!')
