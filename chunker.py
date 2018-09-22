import sklearn_crfsuite
from sklearn.externals import joblib
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, ClassifierMixin

class Chunker(BaseEstimator, ClassifierMixin):
    def __init__(self, verbose = True):
       self.crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=200,
                all_possible_transitions=True,
                verbose=verbose
            )

       self.fitted = False

    def new_tags(self, tags):
        bad_tags = ['anim', 'inan', 'ms-f', 'Sgtm', 'Pltm', 'Fixd', 'Abbr', 'Name', 'Surn', 'Patr', 'Geox', 'Orgn',
                    'Trad',
                    'Subx', 'Apro', 'Anum', 'Poss', 'V-ey', 'V-oy', 'Cmp2', 'V-ej', 'Impx', 'Mult', 'Refl', 'incl',
                    'excl',
                    'Infr', 'Slng', 'Arch', 'Litr', 'Erro', 'Dist', 'Ques', 'Dmns', 'Prnt', 'V-be', 'V-en', 'V-ie',
                    'V-bi',
                    'Fimp', 'Prdx', 'Coun', 'Coll', '-sh', 'Af-p', 'Inmx', 'Vpre', 'Anph', 'Init', 'Adjx', 'Ms-f']
        clean_tags = []
        clean_sent_tags = []
        for ta in tags:
            clean_tags = []
            if ta not in bad_tags:
                clean_sent_tags.append(ta)
        return clean_sent_tags

    def load_from_path(self, path, with_y = True):
        X = [list()]
        if with_y:
            y = [list()]
        if with_y:
            num_spaces = 2
        else:
            num_spaces = 1
        with open(path, 'r', encoding='utf-8') as f:
            for num, line in enumerate(f):
                if line.count(" ") == num_spaces:
                    parts = line[:-1].split()
                    word = parts[0]
                    tags = parts[1].replace(" ", ",").split(",")
                    tags = self.new_tags(tags)
                    X[-1].append({'word': word, 'tags': tags})
                    if with_y:
                        expected_output = parts[2]
                        y[-1].append(expected_output)
                elif len(line) < 2:
                    X.append([])
                    if with_y:
                        y.append([])
                else:
                    raise ValueError("Error in file {path}: "
                                     "Line {line} at #{num} contains "
                                     "{i} space chars. It should either be emtpy "
                                     "Or contain {num_spaces}.".format(
                        path=path, line=line, num=num,
                        i=line.count(""), num_spaces = num_spaces
                    ))
        if with_y:
            return X, y
        else:
            return X

    def load_model(self, path_to_crf):
        self.crf = joblib.load(path_to_crf)
        self.fitted = True
        return self

    def fit(self, path_to_train, path_to_save_model = None):
        X, y = self.load_from_path(path_to_train)
        self.crf.fit(X, y)
        self.fitted = True

        if path_to_save_model is not None:
            joblib.dump(self.crf, path_to_save_model)

        return self.crf

    def predict(self, X = None, path_to_data = None, with_y = True):
        if not self.fitted:
            raise NotFittedError

        if X is None and path_to_data is None:
            raise ValueError("Either X or path_to_data must be specified")
        if X is not None and path_to_data is not None:
            raise ValueError("Either X or path_to_data (not both) must be specified")

        if X is not None:
            return self.crf.predict(X)
        if path_to_data is not None:
            if with_y:
                X, y = self.load_from_path(path_to_data, with_y = with_y)
            else:
                X = self.load_from_path(path_to_data, with_y = with_y)

            return self.crf.predict(X)


    def fit_transform(self, path_to_data):
        return self.fit(path_to_data).transform(path_to_data)

if __name__ == '__main__':
    pass
