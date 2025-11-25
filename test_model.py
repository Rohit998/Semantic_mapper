from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("fine_tuned_model")

s1 = "Accurately presents an organized summary of a patient case verbally and in writing"
s2 = "Communicate effectively with colleagues within one’s profession or specialty, other health professionals, and health related agencies"

score = util.cos_sim(model.encode(s1, convert_to_tensor=True),
                     model.encode(s2, convert_to_tensor=True)).item()

print("Similarity Score:", round(score, 2))
