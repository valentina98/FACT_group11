import pickle
import nltk
import gzip
from nltk.corpus import framenet as fn
nltk.download('framenet_v17')

def save_chunk(chunk, chunk_index):
    filename = f'framenet_sentences_{chunk_index}.pkl.gz'
    with gzip.open(filename, 'wb') as file:
        pickle.dump(chunk, file)
    print(f"Saved chunk {chunk_index}")

def get_framenet_sentences(chunk_size=100):
    sentences_with_frames = {}
    frame_count = 0
    chunk_index = 1

    for frame in fn.frames():
        frame_count += 1
        frame_name = frame.name
        sentences_with_frames[frame_name] = {'positive': [], 'negative': []}

        for lu_name, lu in frame.lexUnit.items():
            lu_id = lu.ID
            lu_object = fn.lu(lu_id)
            for exemplar in lu_object.exemplars:
                sentence_text = exemplar.text
                sentences_with_frames[frame_name]['positive'].append(sentence_text)

        if frame_count % chunk_size == 0:
            save_chunk(sentences_with_frames, chunk_index)
            sentences_with_frames = {}
            chunk_index += 1

    # Saving the last chunk
    if sentences_with_frames:
        save_chunk(sentences_with_frames, chunk_index)

def main():
    get_framenet_sentences()

if __name__ == "__main__":
    main()

