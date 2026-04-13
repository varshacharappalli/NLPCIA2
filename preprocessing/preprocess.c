#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TEXT     100000
#define MAX_SENT     500
#define MAX_SENT_LEN 2000
#define MAX_TOK_LEN  200

/* ── globals ──────────────────────────────────────────────── */
static char text_buf[MAX_TEXT + 1];
static char sentences[MAX_SENT][MAX_SENT_LEN];
static int  num_sentences = 0;
static int  total_tokens  = 0;

/* ── abbreviation table ───────────────────────────────────── */
static const char *ABBREVS[] = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "vs", "etc", "inc", "ltd", "corp", "fig", "no",
    "st", "ave", "dept", "est", "approx", "govt",
    NULL
};

static int is_abbreviation(const char *word_end, const char *text_start) {
    /* word_end points to the char BEFORE the dot.
       Walk backward to find the start of the word. */
    const char *p = word_end;
    while (p > text_start && isalpha((unsigned char)*(p-1)))
        p--;
    /* Extract word (lowercase) */
    char word[64];
    int len = (int)(word_end - p + 1);
    if (len <= 0 || len >= 63) return 0;
    for (int i = 0; i < len; i++)
        word[i] = (char)tolower((unsigned char)p[i]);
    word[len] = '\0';

    for (int i = 0; ABBREVS[i]; i++) {
        if (strcmp(word, ABBREVS[i]) == 0) return 1;
    }
    return 0;
}

static int is_decimal(const char *dot_pos, const char *text_start) {
    /* Check if dot is part of decimal number: digit.digit */
    if (dot_pos == text_start) return 0;
    if (!isdigit((unsigned char)*(dot_pos - 1))) return 0;
    if (!isdigit((unsigned char)*(dot_pos + 1))) return 0;
    return 1;
}

/* ── segmentation ─────────────────────────────────────────── */
static void segment_sentences_rules(const char *text) {
    int len = (int)strlen(text);
    int sent_start = 0;
    num_sentences = 0;

    for (int i = 0; i < len && num_sentences < MAX_SENT - 1; i++) {
        char c = text[i];

        if (c == '.' || c == '!' || c == '?') {
            /* Check if this is a sentence boundary */
            int is_boundary = 0;

            if (c == '!' || c == '?') {
                /* Always a boundary if followed by whitespace + uppercase or end */
                int j = i + 1;
                while (j < len && text[j] == ' ') j++;
                if (j >= len || (isspace((unsigned char)text[j]) ||
                    isupper((unsigned char)text[j]))) {
                    is_boundary = 1;
                }
            } else {
                /* c == '.' */
                /* Not a boundary if abbreviation */
                if (is_abbreviation(text + i - 1, text + sent_start)) {
                    is_boundary = 0;
                }
                /* Not a boundary if decimal number */
                else if (i + 1 < len && is_decimal(text + i, text + sent_start)) {
                    is_boundary = 0;
                }
                /* Not a boundary if followed by lowercase or digit */
                else {
                    int j = i + 1;
                    while (j < len && text[j] == ' ') j++;
                    if (j < len && (islower((unsigned char)text[j]) ||
                                    isdigit((unsigned char)text[j]))) {
                        is_boundary = 0;
                    } else if (j < len && isupper((unsigned char)text[j])) {
                        is_boundary = 1;
                    } else if (j >= len) {
                        is_boundary = 1;
                    }
                }
            }

            if (is_boundary) {
                /* Extract sentence from sent_start to i (inclusive) */
                int slen = i - sent_start + 1;
                if (slen > 0 && slen < MAX_SENT_LEN) {
                    /* Trim leading whitespace */
                    int s = sent_start;
                    while (s < i && isspace((unsigned char)text[s])) s++;
                    slen = i - s + 1;
                    if (slen > 0) {
                        strncpy(sentences[num_sentences], text + s, (size_t)slen);
                        sentences[num_sentences][slen] = '\0';
                        num_sentences++;
                    }
                }
                /* Skip whitespace after punctuation */
                i++;
                while (i < len && isspace((unsigned char)text[i])) i++;
                sent_start = i;
                i--; /* loop will increment */
            }
        }
    }

    /* Remaining text after last boundary */
    if (sent_start < len) {
        int s = sent_start;
        while (s < len && isspace((unsigned char)text[s])) s++;
        int slen = len - s;
        /* Trim trailing whitespace */
        while (slen > 0 && isspace((unsigned char)text[s + slen - 1])) slen--;
        if (slen > 0 && slen < MAX_SENT_LEN && num_sentences < MAX_SENT) {
            strncpy(sentences[num_sentences], text + s, (size_t)slen);
            sentences[num_sentences][slen] = '\0';
            num_sentences++;
        }
    }
}

static void segment_sentences_newline(const char *text) {
    int len = (int)strlen(text);
    num_sentences = 0;
    int start = 0;

    for (int i = 0; i <= len && num_sentences < MAX_SENT; i++) {
        if (i == len || text[i] == '\n') {
            int s = start;
            int slen = i - s;
            /* Trim leading */
            while (slen > 0 && isspace((unsigned char)text[s])) { s++; slen--; }
            /* Trim trailing */
            while (slen > 0 && isspace((unsigned char)text[s + slen - 1])) slen--;
            if (slen > 0 && slen < MAX_SENT_LEN) {
                strncpy(sentences[num_sentences], text + s, (size_t)slen);
                sentences[num_sentences][slen] = '\0';
                num_sentences++;
            }
            start = i + 1;
        }
    }
}

/* ── transformations ──────────────────────────────────────── */
static void apply_lowercase(char *s) {
    for (int i = 0; s[i]; i++)
        s[i] = (char)tolower((unsigned char)s[i]);
}

static void apply_remove_punct(char *s) {
    for (int i = 0; s[i]; i++) {
        char c = s[i];
        /* Keep apostrophes and hyphens */
        if (c == '\'' || c == '-') continue;
        if (ispunct((unsigned char)c))
            s[i] = ' ';
    }
}

/* ── tokenizer ────────────────────────────────────────────── */
static int tokenize_sentence(const char *s, char tokens[][MAX_TOK_LEN]) {
    int count = 0;
    int len = (int)strlen(s);
    int i = 0;

    while (i < len && count < 200) {
        /* Skip whitespace */
        while (i < len && isspace((unsigned char)s[i])) i++;
        if (i >= len) break;
        /* Read token */
        int start = i;
        while (i < len && !isspace((unsigned char)s[i])) i++;
        int tlen = i - start;
        if (tlen > 0 && tlen < MAX_TOK_LEN) {
            strncpy(tokens[count], s + start, (size_t)tlen);
            tokens[count][tlen] = '\0';
            count++;
        }
    }
    return count;
}

/* ── main ─────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    int do_lowercase  = 1;
    int do_punct      = 1;
    int do_seg_rules  = 1;

    /* Parse flags */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-lowercase") == 0) do_lowercase = 0;
        else if (strcmp(argv[i], "--no-punct")    == 0) do_punct     = 0;
        else if (strcmp(argv[i], "--no-segment")  == 0) do_seg_rules = 0;
    }

    /* Read stdin */
    int text_len = 0;
    int c;
    while ((c = getchar()) != EOF && text_len < MAX_TEXT) {
        text_buf[text_len++] = (char)c;
    }
    text_buf[text_len] = '\0';

    /* Segment */
    if (do_seg_rules)
        segment_sentences_rules(text_buf);
    else
        segment_sentences_newline(text_buf);

    /* Process each sentence */
    for (int i = 0; i < num_sentences; i++) {
        if (do_lowercase) apply_lowercase(sentences[i]);
        if (do_punct)     apply_remove_punct(sentences[i]);
    }

    /* SENTENCES section */
    printf("SENTENCES_START\n");
    for (int i = 0; i < num_sentences; i++)
        printf("%s\n", sentences[i]);
    printf("SENTENCES_END\n");

    /* TOKENS section */
    char tokens[200][MAX_TOK_LEN];
    printf("TOKENS_START\n");
    for (int i = 0; i < num_sentences; i++) {
        int nt = tokenize_sentence(sentences[i], tokens);
        for (int t = 0; t < nt; t++) {
            if (t > 0) printf(" ");
            printf("%s", tokens[t]);
        }
        printf("\n");
        total_tokens += nt;
    }
    printf("TOKENS_END\n");

    /* STATS section */
    printf("STATS_START\n");
    printf("num_sentences:%d\n", num_sentences);
    printf("num_tokens:%d\n",    total_tokens);
    printf("STATS_END\n");

    return 0;
}
