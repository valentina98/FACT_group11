import pickle
import nltk
from nltk.corpus import framenet as fn
nltk.download('framenet_v17')

def get_framenet_sentences():
    sentences_with_frames = {}
    all_sentences = []
    print(len(fn.frames()))
    for frame in fn.frames():
        frame_name = frame.name
        sentences_with_frames[frame_name] = {'positive': [], 'negative': []}

        for lu_name, lu in frame.lexUnit.items():
            lu_id = lu.ID
            lu_object = fn.lu(lu_id)
            for exemplar in lu_object.exemplars:
                sentence_text = exemplar.text
                sentences_with_frames[frame_name]['positive'].append(sentence_text)
                #print("Frame name: " +  str(frame_name) + " text: " + sentence_text)
                all_sentences.append(sentence_text)
    for frame, data in sentences_with_frames.items():
        negative_samples = [s for s in all_sentences if s not in data['positive']]
        sentences_with_frames[frame]['negative'] = negative_samples

    with open('framenet_sentences.pkl', 'wb') as file:
        pickle.dump(sentences_with_frames, file)

def main():
    get_framenet_sentences()

if __name__ == "__main__":
    main()

