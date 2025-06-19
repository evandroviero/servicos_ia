 
import google.generativeai as genai

key = 'AIzaSyAoO9-rj5vPtSIL_NZCaV10YGJUAeCeMz0'

genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Gera o conteúdo
response = model.generate_content("Escreva um poema curto sobre a beleza do universo.")

print(response.text)