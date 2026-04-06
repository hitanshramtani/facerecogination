
"""ArcFace is one of the famous deep face recognition methods nowadays. The main feature of ArcFace is applying an 
Additive Angular Margin Loss to enforce the intra-class(same person) compactness of embedding space and inter-class(others) discrepancy. 
Before ArcFace arrived, many proposed methods used the softmax loss as a classification loss in deep face recognition. However, there 
are some drawbacks to using only the softmax loss. One of them is that the softmax loss doesn’t optimize the feature embedding to 
constrain higher similarity between intra-class samples and diversity for inter-class samples, which means the boundaries between people 
are sometimes ambiguous, and it deteriorates the model performance. So, Additive Angular Margin Loss to obtain 
further improvement for the discriminative power of face recognition."""
# LAAM = (e^scos(θ_yi​​+m))/(e^scos(θ_yi​​+m)​+∑j=1,j=N,j!=yi{e^scos(θ_j​)})

# Dataset Recieved successfully

# Dataset label correction
"""
label were not correct , corrected them by removing extra spaces and unwanted characters and
defining them into enroll_no+ Full Name
"""

# Dataset Augmentation
"""
done augmentation of dataset and created a new folder with name 'EO_augment_pics' and 'IO_augment_pics'
and added 5 augment pics of each student in their respective folder
"""

# face bank and label created successfully
"""
FOR EO
Student1
Added 'BTEO24O1001 AASHU DHAKAD' with 5 samples.
Student2
Added 'BTEO24O1003 AJAY PRAJAPATI' with 5 samples.
....
Student46
Added 'BTEO24O1054 Yashaswi Bhargava' with 5 samples.
Student47
Added 'BTEO24O1055 ANURAG SHARMA' with 5 samples.

For IO
Student1
Added 'BTIO24O1001 AARADHYA PURANIK' with 5 samples.
Student2
Added 'BTIO24O1002 ABHAY PALWAY' with 5 samples.
....
Student69
Added 'BTIO24O1077 YASH SAHU' with 5 samples.
Student70
Added 'BTIO24O1078 YATHARTH GUPTA' with 5 samples.


Face bank created successfully!"""

# main app creation start 
"""
created detect face , get embedding, recognize face, mark attendance function
and created a main loop for attendance marking
and face recognition
but
the problem started over when the app was misbehaving and not detecting the face properly and recoginize each face with aashu dhakad

"""

# Debug Stage
"""
Made a test file  start debugging over there each values were printed to get 
problem identified in face bank when printed face it show each image has same embedding
"""

# Problem disscussion and solution
"""
During building face bank i was not using the face detection and alignment function in that i made the embediing as int which was giving
error
face_np = face_tensor.permute(1, 2, 0).int().cpu().numpy() -> face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
"""

# Another problem was face recognition was not working properly and giving wrong names
"""

"""

