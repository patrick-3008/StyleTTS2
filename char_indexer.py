# IPA Phonemizer: https://github.com/bootphon/phonemizer

import string

class BertCharacterIndexer:
    PAD = "P"
    PUNCTUATION = ''.join(sorted(set(';:,.!?¡¿—…"«»“”‘’،؛؟٫٬٪﴾﴿ـ' + string.punctuation)))
    LETTERS_IPA = 'ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘̩ᵻ'
    LATIN_LETTERS = 'abcdefghijklmnopqrstuvwxyz'
    PHONEME_MASK = "M"
    PHONEME_SEPARATOR = " "
    # NOTE: '¤' is a valid 'unknown' character because it is different from all the characters above it. In English PL-BERT, 'U' was used as the unknown character which was not ideal as it was part of the English alphabet
    UNKNOWN='U'
    # Export all symbols:
    symbols = [PAD] + list(PUNCTUATION) + list(LETTERS_IPA) + list(LATIN_LETTERS) + [PHONEME_MASK] + [PHONEME_SEPARATOR] + [UNKNOWN]

    assert len(symbols) == len(set(symbols)) # no duplicates

    def __init__(self):
        self.word_index_dictionary = {symbol: i for i, symbol in enumerate(BertCharacterIndexer.symbols)}

    def __call__(self, text):
        return [self.word_index_dictionary[char] if char in self.word_index_dictionary 
                else self.word_index_dictionary[BertCharacterIndexer.UNKNOWN] for char in text]

class VanillaCharacterIndexer:
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    # Export all symbols:
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

    def __init__(self):
        dicts = {}
        for i in range(len(self.symbols)):
            dicts[self.symbols[i]] = i

        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes