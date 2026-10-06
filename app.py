"""Streamlit app: brain tumour information and MRI classification demo.

Run with:  streamlit run app.py
Educational use only. Not a diagnostic tool.
"""
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_option_menu import option_menu

MODEL_PATH = Path(__file__).parent / "best_cnnmodel_1.h5"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

st.set_page_config(page_title="Brain Tumour Classification")


@st.cache_resource
def load_model():
    """Load the trained CNN once and reuse it across reruns."""
    return tf.keras.models.load_model(MODEL_PATH)


def predict(image, model):
    """Return class probabilities for a PIL image.

    The model was trained on 150x150 images read with OpenCV (BGR channel
    order, pixel values 0-255), so the upload is converted to match.
    """
    image = ImageOps.fit(image.convert("RGB"), IMG_SIZE, Image.LANCZOS)
    bgr = np.asarray(image, dtype="float32")[..., ::-1]
    return model.predict(bgr[np.newaxis, ...], verbose=0)[0]


selected = option_menu(
  menu_title=None,
  options=["Home","Treatment & Diagnosis","Detection"],
  menu_icon="cast",
  orientation="vertical",
)
if selected == "Home":
  st.write("The signs and symptoms of a brain tumor vary greatly and depend on the brain tumor's size, location and rate of growth.General signs and symptoms caused by brain tumors may include:New onset or change in pattern of headaches.Headaches that gradually become more frequent and more severe Unexplained nausea or vomiting Vision problems, such as blurred vision, double vision or loss of peripheral vision Gradual loss of sensation or movement in an arm or a leg Difficulty with balance, Speech difficulties,Feeling very tired Confusion in everyday matters,Difficulty making decisions and Inability to follow simple commands and Personality or behavior changesSeizures, especially in someone who doesn't have a history of seizures Hearing problems")
  st.subheader("Causes:")
  st.write("Brain tumors that begin in the brain Primary brain tumors originate in the brain itself or in tissues close to it, such as in the brain-covering membranes (meninges), cranial nerves,pituitary gland or pineal gland. Primary brain tumors begin when normal cells develop changes (mutations) in their DNA.  A cell's DNA contains the instructions that tell a cell what to do. The mutations tell the cells  to grow and divide rapidly and to continue living when healthy cells would die. The result is a mass of abnormal cells, which forms a tumor. In adults, primary brain tumors are much less common than are secondary brain tumors, in which cancer begins elsewhere and spreads to the brain. Many different types of primary brain tumors exist. Each gets its name from the type of cells involved.Examples include:Gliomas. These tumors begin in the brain or spinal cord and include astrocytomas, ependymomas, glioblastomas, oligoastrocytomas and oligodendrogliomas.Meningiomas. A meningioma is a tumor that arises from the membranes that surround your brain and spinal cord (meninges).Most meningiomas are noncancerous.Acoustic neuromas (schwannomas). These are benign tumors that develop on the nerves that control balance and hearing leading from your inner ear to your brain.Pituitary adenomas. These are tumors that develop in the pituitary gland at the base of the brain.These tumors can affect the pituitary hormones with effects throughout the body.Medulloblastomas. These cancerous brain tumors are most common in children, though they can occur at any age. A medulloblastoma starts in the lower back part of the brain and tends to spread through the spinal fluid.Germ cell tumors. Germ cell tumors may develop during childhood where the testicles or ovaries will form. But sometimes germ cell tumors affect other parts of the body, such as the brain.Craniopharyngiomas. These rare tumors start near the brain's pituitary gland, which secretes hormones that control many body functions. As the craniopharyngioma slowly grows, it can affect the pituitary gland and other structures near the brain.Cancer that begins elsewhere and spreads to the brainSecondary (metastatic) brain tumors are tumors that result from cancer that starts elsewhere in your body and then spreads (metastasizes) to your brain.Secondary brain tumors most often occur in people who have a history of cancer. Rarely, a metastatic brain tumor may be the first sign of cancer that began elsewhere in your body.In adults, secondary brain tumors are far more common than are primary brain tumors.Any cancer can spread to the brain, but common types include:Breast cancer,Colon cancer,Kidney cancer,Lung cancer,Melanoma.Risk factorsIn most people with primary brain tumors, the cause of the tumor isn't clear. But doctors have identified some factors that may increase your risk of a brain tumor.Risk factors include:Exposure to radiation. People who have been exposed to a type of radiation called ionizing radiation have an increased risk of brain tumor. Examples of ionizing radiation include radiation therapy used to treat cancer and radiation exposure caused by atomic bombs.Family history of brain tumors. A small portion of brain tumors occurs in people with a family history of brain tumors or a family history of genetic syndromes that increase the risk of brain tumors.")

#####

if selected == "Treatment & Diagnosis":
  st.header("Treatment")
  tab1, tab2, tab3, tab4, tab5 = st.tabs(["Surgery", "Radiation", "Radiosurgery", "Chemotherapy", "Targeted therapy"])

  with tab1:
   st.header("Surgery")
   #st.image("https://static.streamlit.io/examples/cat.jpg")
   st.write("If the brain tumor is located in a place that makes it accessible for an operation,  your surgeon will work to remove as much of the brain tumor as can be done safely. Some brain tumors are small and easy to separate from surrounding brain tissue, which makes complete surgical removal possible.   Other brain tumors can't be separated from surrounding tissue or they're located near sensitive areas in your brain, making surgery risky.  In these situations, your doctor removes as much of the tumor as is safe.Even removing a portion of the brain tumor may help reduce your signs and symptoms. Surgery to remove a brain tumor carries risks, such as infection and bleeding. Other risks may depend on the part of your brain where your tumor is located. For instance, surgery on a tumor near nerves that connect to your eyes may carry a risk of vision loss.")

  with tab2:
   st.header("Radiation Therapy")
   #st.image("https://static.streamlit.io/examples/dog.jpg")
   st.write("Radiation therapy uses high-energy beams, such as X-rays or protons, to kill tumor cells. Radiation therapy can come from a machine outside your body (external beam radiation), or, very rarely, radiation can be placed inside your body  close to your brain tumor (brachytherapy). External beam radiation can focus just on the area of your brain where the tumor is located, or it can be applied to your entire brain (whole-brain radiation).  Whole-brain radiation is most often used to treat cancer that spreads to the brain from some other part of the body and forms multiple tumors in the brain. Traditionally, radiation therapy uses X-rays, but a newer form of this treatment uses proton beams. Proton beam therapy allows doctors to control the radiation more precisely. It may be helpful for treating brain tumors in children and tumors that are very close to sensitive areas of the brain.  Proton beam therapy isn't as widely available as traditional X-ray radiation therapy.Side effects of radiation therapy depend on the type and dose of radiation you receive.  Common side effects during or immediately following radiation include fatigue, headaches, memory loss, scalp irritation and hair loss.")
  with tab3:
   st.header("Radiosurgery")
   #st.image("https://static.streamlit.io/examples/owl.jpg")
   st.write("Stereotactic radiosurgery is not a form of surgery in the traditional sense. Instead, radiosurgery uses multiple beams of radiation to give a highly focused form of   radiation treatment to kill the tumor cells in a very small area. Each beam of radiation isn't particularly powerful, but the point where all the beams meet — at the    brain tumor — receives a very large dose of radiation to kill the tumor cells.   There are different types of technology used in radiosurgery to deliver radiation to treat brain tumors, such as a Gamma Knife or linear accelerator.  Radiosurgery is typically done in one treatment, and usually you can go home the same day.")

  with tab4:
   st.header("Chemotherapy")
   #st.image("https://static.streamlit.io/examples/owl.jpg")
   st.write("Chemotherapy uses drugs to kill tumor cells. Chemotherapy drugs can be taken orally in pill form or injected into a vein (intravenously).  The chemotherapy drug used most often to treat brain tumors is temozolomide (Temodar). Other chemotherapy drugs may be recommended depending on the type of cancer. Chemotherapy side effects depend on the type and dose of drugs you receive. Chemotherapy can cause nausea, vomiting and hair loss.Tests of your brain tumor cells can determine whether chemotherapy will be helpful for you. The type of brain tumor you have also is helpful in determining whether to recommend chemotherapy.")

  with tab5:
   st.header("Targeted drug therapy")
   #st.image("https://static.streamlit.io/examples/owl.jpg")
   st.write("Targeted drug treatments focus on specific abnormalities present within cancer cells. By blocking these abnormalities, targeted drug treatments can cause cancer cells to die.  Targeted therapy drugs are available for certain types of brain tumors, and many more arebeing studied in clinical trials. Your doctor may have your tumor cells tested to see whether targeted therapy is likely to be an effective treatment for your brain tumor.")

  st.subheader("Rehabilation after Treatment")
  st.write("Because brain tumors can develop in parts of the brain that control motor skills, speech, vision and thinking, rehabilitation may be a necessary part of recovery.Depending on your needs, your doctor may refer you to: Physical therapy to help you regain lost motor skills or muscle strength Occupational therapy to help you get back to your normal daily activities, including work, after a brain tumor or other illness Speech therapy with specialists in speech difficulties (speech pathologists) to help if you have difficulty speaking. Tutoring for school-age children to help kids cope with changes in their memory and thinking after a brain tumor")

  st.header("Diagnosis")
  st.write("A neurological exam. A neurological exam may include, among other things, checking your vision, hearing, balance, coordination, strength and reflexes.Difficulty in one or more areas may provide clues about the part of your brain that could be affected by a brain tumor. Imaging tests. Magnetic resonance imaging (MRI) is commonly used to help diagnose brain tumors. Sometimes a dye is injected through a vein in your arm during your MRI study.A number of specialized MRI scan components — including functional MRI, perfusion MRI and magnetic resonance spectroscopy — may help your doctor evaluate the tumor and plan treatment.Sometimes other imaging tests are recommended in certain situations, including computerized tomography (CT) and positron emission tomography (PET).Collecting and testing a sample of abnormal tissue (biopsy).A biopsy can be performed as part of an operation to remove the brain tumor, or a biopsy can be performed using a needle. A stereotactic needle biopsy may be done for brain tumors in hard to reach areas or very sensitive areas within your brain that might be damaged by a more extensive operation. Your neurosurgeon drills a small hole into your skull. A thin needle is then inserted through the hole. Tissue is removed using the needle, which is frequently guided by CT or MRI scanning. The biopsy sample is then viewed under a microscope to determine if it's cancerous or benign.  Sophisticated laboratory tests can give your doctor clues about your prognosis and your treatment options. Studying your biopsy sample and determining exactly which type of brain tumor you have is a complex process. If you're uncertain about your diagnosis, consider seeking a second opinion at a medical center where many brain biopsies are evaluated every year.")
  ######

if selected == "Detection":
    st.header("Brain tumour classification")
    st.caption("Educational demo only. This is not a diagnostic tool.")

    file = st.file_uploader("Please upload a brain MRI scan file", type=["jpg", "jpeg", "png"])
    if file is None:
        st.text("Please upload an image file")
    elif not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
    else:
        image = Image.open(file)
        st.image(image, use_container_width=True)
        with st.spinner("Model is being loaded..."):
            model = load_model()
        probabilities = predict(image, model)
        best = int(np.argmax(probabilities))
        st.write(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(CLASS_NAMES[best], 100 * float(probabilities[best]))
        )
        st.bar_chart(dict(zip(CLASS_NAMES, map(float, probabilities))))
