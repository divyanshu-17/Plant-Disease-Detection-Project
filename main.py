import streamlit as st
import tensorflow as tf 
import numpy as np


#Tensorflow model prediction
def model_prediction(test_image):
    model=tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128)) 
    input_arr = tf.keras.preprocessing.image.img_to_array (image)
    input_arr = np.array([input_arr]) #Convert single image to a batch 
    prediction=model.predict(input_arr) 
    result_index = np.argmax(prediction) 
    return result_index

#Sidebarw
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Diseases Recognition","Contact"])

#Homepage
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path= "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)
     # Add the interactive image gallery here
    st.image(["test\TomatoEarlyBlight1.JPG", "test\AppleCedarRust1.JPG", "test\CornCommonRust1.JPG","test\PotatoEarlyBlight4.JPG"], caption=["Disease 1", "Disease 2", "Disease 3","Disease 4"], width=300 )

    st.markdown("""
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page...
    """)

#Aboutpage
# About Page
elif(app_mode == "About"):
    st.header("About")
    st.markdown("""
    ### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo. This dataset consists of about 11.5K RGB images of healthy and diseased crop leaves which are categorized into 38 different classes. 

    The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.

    ### Content
    1. Train (5876 images)  
    2. Valid (5776 images)  
    3. Test (33 images)
    """)

    # Adding expander for Project Goals
    with st.expander("Project Goals"):
        st.write("""
        Our mission is to improve plant disease detection using cutting-edge machine learning algorithms.
        This project aims to provide an efficient system to detect various diseases affecting plants 
        and crops to ensure better yield and healthier plants.
        """)

    # Team Section
    st.markdown("### Meet the Team")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("harsh.jpg", width=130)
        st.write("**Harsh Awasthi**")

    with col2:
        st.image("divyanshu.jpg", width=140)
        st.write("**Divyanshu Ranjan**")

    with col3:
        st.image("awanish.jpg", width=130)
        st.write("**Awanish Bhatt**")

    # Fun Facts Section
    st.markdown("### Fun Facts")
    st.write("🌱 Did you know that plants can suffer from over 1,000 different diseases?")

    # Dataset Distribution Bar Chart
    st.markdown("### Dataset Distribution")
    st.bar_chart({
        "Dataset": [5876, 5776, 33]
    }, use_container_width=True)


#Prediction page
elif(app_mode=="Diseases Recognition"):
    st.header("Diseases Recognition")
    test_image= st.file_uploader("Choose an image:")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    #predict button
    if(st.button("Predict")):
        st.snow()
        with st.spinner("Please Wait.."):
            st.write("Our Prediction")
            result_index =model_prediction(test_image)
            #Define class
            class_name =['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)__Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape__Esca(Black_Measles)',
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange__Haunglongbing(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,bell__Bacterial_spot',
    'Pepper,bell__healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

elif(app_mode=="Contact"):
    st.header("Contact Us")
    st.markdown("### Contact Us")
    st.write("""
    If you have any questions or queries, feel free to reach out to us! You can contact us at:
    """)

    # Use of icons for contact details
    st.markdown("📧 **Email:** harshawasthi6307@gmail.com")
    st.markdown("📞 **Phone:** 9026177351")
    st.markdown("📱 **Alternate Phone:** 6307323261")

    # Add a button for feedback or inquiries
    if st.button("Send Us a Message"):
        st.write("Thank you! Your message has been sent.")









        

