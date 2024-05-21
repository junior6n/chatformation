#!/usr/bin/env python
# coding: utf-8

# ## Librairie a importer

# In[1]:


# Pour le webscraping
from bs4 import BeautifulSoup
import requests
import streamlit as st
import numpy as np

# Pour l'apprentissage automatique
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
from sentence_transformers import SentenceTransformer


# ## Model d'extraction des mots cle dans un phrase
# Ce model sera utiliser pour extrait les mot cle dans la description donnee par l'utilisateur

# In[2]:


# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])


# In[5]:


# Load pipeline
def mot_cle(description):
  model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
  extractor = KeyphraseExtractionPipeline(model=model_name)

  # Inference
  des_text = description

  return extractor(des_text)

  


# ## Extraction des donnee dans les different site de formation

# In[6]:




# 
# ### Extraction des donnee de quelque formation du site Coursera
# 
# Ce scripte permet d'extrait les donnee dans le site ***coursera.org***, selon une recherche sur la formation ou le cours en ligne que vous voulez,  pour chaque formation proposer ca vas extrait:
# - Le Titre de la formation
# - Les Competences a acquerrir
# - La description de la formation
# - Le lien de la formation
# 

# In[16]:

def coursera_scrap(keyphrases, formation_data):
  print("Scraping OK")
  formation_data['coursera.org'] = []

  nombre_page = 1      #le nombre de page ou qu'on doit extrait les donnee, chaque page propose au maximum les information sur 12 formation
  i = 0     # le conteur initial, de la premiere boucle, qui control le nombre de page choisir

  query_ans = '%20'.join(keyphrases)      # la combinaison des mot cle en un seul phrase qui va etre utiliser pour la recherche

  while(i<nombre_page):
      url = f"https://www.coursera.org/search?query={query_ans}&page={i+1}&sortBy=BEST_MATCH"

      result = requests.get(url)

      soup = BeautifulSoup(result.text, "html.parser")

      ul_tag = soup.find('ul', {'class': "cds-9 css-reop8o cds-10"})

      # verifie si les valeur des balise html qu'on doit recuperer est Null, si oui ca doit recommencer la boucle a zero
      if(ul_tag == None):
          continue

      # iterer sur les balise qui contient les information sur les different formation
      list_li_tag = ul_tag.find_all('li')

      for course_info in list_li_tag:
          # l'image de la formation
          img_tag = course_info.find('img') # La balise img qui contient les information sur l'image
          images = img_tag.attrs['src'] # src c'est l'attribut qui a pout valeur le lien de l'image
          # print(images)
          img = f"https://{images.split(':')[2]}".replace("////", "//")
          if(img == None):
              img = ""

          # le titre du cours
          title_tag = course_info.find('a') # la balise ancre qui contient le titre et le lien de la formation
          title = title_tag.text # titre de la formation
          if (title == None):
              title = ""

          # competence a acquerrir
          comp_tag = course_info.find('div',{'class':"cds-CommonCard-bodyContent"})
          
          comp = comp_tag.text.split(":")[1]
          if(comp_tag == None):
             comp = ""
            
          # le lien de la formation
          link = f"https://www.coursera.org{title_tag.attrs['href']}"
          if (link == None):
              link=""

          # Ce que vous aller apprendre

          # ressayer trois fois
          j = 0   # counteur
          descriptiontions = [""]
          while(j<3):
            soup2 = BeautifulSoup(requests.get(link).text, 'html.parser')
            desc_ul_tag = soup2.find('ul', {'class': 'cds-9 css-7avemv cds-10'})

            if(desc_ul_tag == None):
              j = j+1
              continue  # pour ressayer apres que la valeur est None

            descs_li_tags = desc_ul_tag.find_all('li')

            descs_list = []
            for dl in descs_li_tags:
                descs_list.append(f"- {dl.text}")
            descriptions = '\n'.join(descs_list)
            if(descriptions == None):
                descriptions = ""
            break   # pour arreter

          # stockage des Information sur la formation
          form = {'image': img, 'titre': title, 'competence': comp,  'description': descriptions, 'lien': link}
          formation_data['coursera.org'].append(form)

          # print('Titre: ', title)
          # if(comp_tag != None):
          #  print('Competence que vous acquererz: ', comp)
          # print('What you will learn: \n', descriptions)
          # print('lien: ', link)
          # print('Nom du Site: Coursera.org')
          # print(10*'-----', '\n')

      i = i + 1

      return formation_data





# In[20]:


def convert_text(formation_data):
  print("Conversion text OK")
  form_info = []
  for i in range(len(formation_data['coursera.org'])):
    ftext = '\n'.join(list(formation_data['coursera.org'][i].values())[1:-1])
    form_info.append(f"""{ftext}""")
  return form_info


# ## Model de verification de la similarites des phrases
#  Ce model sera utilises pour verifier la similarites  de la description donnee pas l'utilisateur avec la description de nos different formation propose

# In[21]:


def cos_sim(x, y):
  """
  input: Two numpy arrays, x and y
  output: similarity score range between 0 and 1
  """
    #Taking dot product for obtaining the numerator
  numerator = np.dot(x, y)

    #Taking root of squared sum of x and y
  x_normalised = np.sqrt(np.sum(x**2))
  y_normalised = np.sqrt(np.sum(y**2))

  denominator = x_normalised * y_normalised
  cosine_similarity = numerator / denominator
  return cosine_similarity


def similarites(des_text, form_info):
  print("Similarites OK")
  # sentences = ["a cat", "a cat", 'dog']

  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  #embeddings = model.encode(sentences)
  embeddings = model.encode([f'{des_text}'] + form_info)

  similarities = {}

  for i in range(1, len(embeddings)):
    similarities[i-1] = cos_sim(embeddings[0], embeddings[i])

  def sort_simi(x):
    return similarities[x]

  sort_forms = sorted(similarities, key=sort_simi, reverse=True)
  return sort_forms


# In[22]:



def resultat(sort_forms, formation_data):
  print("resultat OK")
  f, count, max = formation_data['coursera.org'], 0, 10
  result = []
  for i in sort_forms:
    result.append(f[i])
    count = count + 1
    if(count==max): break

  return result


def main():
  st.title("Chatbot de propostion de formation")

  form = st.form('description')
  desc = form.text_area("Entrer la description de la formation  (s'il vous plait en anglais)", placeholder="Entrer la description ici")
  submitted = form.form_submit_button("Submit")
  # if submitted:
  #     st.write("Ok")
  if submitted:
    formation_data = {}
    formation_data = coursera_scrap(mot_cle(desc), formation_data)
    sort_forms=similarites(desc, convert_text(formation_data))
    result = resultat(sort_forms, formation_data)
    l = len(result)
    i = 0
    while(i<l):
      col_n = 2
      if (l-1-i==0):
        col_n = 1
      c_multi = st.container()
      with c_multi.container():
        cols = st.columns(col_n, gap="medium")
        for col in cols:
          with col:
            st.markdown(f"**{result[i]['titre']}**")
            st.image(f"{result[i]['image']}".replace("////", "//"))
            st.caption(result[i]['competence'])
            st.caption(result[i]['description'])
            st.link_button("link", result[i]['lien'])
          i = i + 1
        st.divider()

      

      #  st.json(i
      #  st.write("\n")

if __name__ == "__main__":
   main()
