import pandas as pd

df_gita = pd.read_csv("Bhagwad_Gita/Bhagwad_Gita_Verses_English_augmented_appended.csv")
df_patanjali = pd.read_csv("Patanjali_Yoga_Sutras/Patanjali_Yoga_Sutras_Verses_English_Questions_augmented_appended.csv")

df_patanjali['speaker'] = 'Rishi Patanjali'

df_gita['book'] = 'Bhagwad Gita'
df_patanjali['book'] = 'Patanjali Yoga Sutras'

columns_order = ['chapter', 'verse', 'speaker', 'sanskrit', 'translation', 'question', 'augmented_response', 'book']
df_gita = df_gita[columns_order]
df_patanjali = df_patanjali[columns_order]

df_concatenated = pd.concat([df_gita, df_patanjali], axis=0, ignore_index=True)

df_concatenated.to_csv("Merged_Bhagwad_Gita_and_Patanjali_Yoga_Sutras.csv", index=False)

print("Dataframes merged and saved successfully.")
