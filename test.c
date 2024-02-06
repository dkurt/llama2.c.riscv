#define TESTING
#include "run.c"

#include <riscv_vector.h>

void assert_eq(int a, int b) {
    if (a != b) {
        printf("Assertion failed: %d != %d\n", a, b);
        exit(EXIT_FAILURE);
    }
}

void test_prompt_encoding(Tokenizer* tokenizer, char* prompt, int* expected_tokens, int num_expected_tokens) {
    // encode
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    int num_prompt_tokens = 0; // the total number of prompt tokens
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    #if VERBOSITY == 1
    // print maybe
    printf("expected tokens:\n");
    for (int i = 0; i < num_expected_tokens; i++) printf("%d ", expected_tokens[i]);
    printf("\n");
    printf("actual tokens:\n");
    for (int i = 0; i < num_prompt_tokens; i++) printf("%d ", prompt_tokens[i]);
    printf("\n");
    #endif

    // verify
    assert_eq(num_prompt_tokens, num_expected_tokens);
    for (int i = 0; i < num_prompt_tokens; i++) {
        assert_eq(prompt_tokens[i], expected_tokens[i]);
    }

    #if VERBOSITY == 1
    printf("OK\n");
    printf("---\n");
    #endif
    free(prompt_tokens);
}

void test_prompt_encodings() {
    // let's verify that the Tokenizer works as expected

    char *tokenizer_path = "tokenizer.bin";
    int vocab_size = 32000;
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, vocab_size);

    // test 0 (test the empty string) (I added this as a simple case)
    char *prompt0 = "";
    int expected_tokens0[] = {1};
    test_prompt_encoding(&tokenizer, prompt0, expected_tokens0, sizeof(expected_tokens0) / sizeof(int));

    // the tests below are taken from the Meta Llama 2 repo example code
    // https://github.com/facebookresearch/llama/blob/main/example_text_completion.py
    // and the expected tokens come from me breaking in the debugger in Python

    // test 1
    char *prompt = "I believe the meaning of life is";
    int expected_tokens[] = {1, 306, 4658, 278, 6593, 310, 2834, 338};
    test_prompt_encoding(&tokenizer, prompt, expected_tokens, sizeof(expected_tokens) / sizeof(int));

    // test 2
    char* prompt2 = "Simply put, the theory of relativity states that ";
    int expected_tokens2[] = {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871};
    test_prompt_encoding(&tokenizer, prompt2, expected_tokens2, sizeof(expected_tokens2) / sizeof(int));

    // test 3
    char* prompt3 = "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ";
    int expected_tokens3[] = {1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871};
    test_prompt_encoding(&tokenizer, prompt3, expected_tokens3, sizeof(expected_tokens3) / sizeof(int));

    // test 4
    char* prompt4 = "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>";
    int expected_tokens4[] = {1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923, 968, 1149};
    test_prompt_encoding(&tokenizer, prompt4, expected_tokens4, sizeof(expected_tokens4) / sizeof(int));

    // memory and file handles cleanup
    free_tokenizer(&tokenizer);
}

// source: https://github.com/opencv/opencv/blob/ae21368eb9b66b448effc60247be8d83056ade80/cmake/checks/cpu_rvv.cpp
int test_rvv()
{
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    unsigned long ptr[2] = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
    vuint8m1_t a = vreinterpret_v_u64m1_u8m1(vle64_v_u64m1(ptr, 2));
    vfloat32m1_t val = vle32_v_f32m1((const float*)(src), 4);
    return (int)vfmv_f_s_f32m1_f32(val);
}



void test_generate(char* prompt, char* checkpoint_path, float temperature, int steps, float topp, const char* expected){
    char *tokenizer_path = "tokenizer.bin";
    unsigned long long rng_seed = 124; // seed rng with time by default

    // parameter validation/overrides
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!

    freopen("output.txt", "wt", stdout);  // redirect output
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    freopen("/dev/tty", "w", stdout);  // resume

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    // Check
    FILE* f = fopen("output.txt", "rt");
    fseek(f, 0, SEEK_END);
    const size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char output[sz];
    fread(output, sizeof(char), sz, f);
    output[sz - 1] = '\0';
    fclose(f);

    int res = strcmp(expected, output);
    if (res != 0) {
        printf("Expected: %s\n\nGenerated: %s\n", expected, output);
    }
    assert_eq(res, 0);
}

int main(int argc, char *argv[]) {
    test_prompt_encodings();
    test_rvv();

    const char* expected = "That was the darkest day of the year. The stars were shining bright in the sky and the birds were singing.\n\
\"Mommy, why is it so dark?\" asked the little girl, pointing out her finger.\n\
\"Well, the sun is setting and it will be a beautiful night,\" replied her mom.\n\
The little girl looked up at the sky and smiled. \"I like it when the sun sets,\" she said.\n\
\"I know, sweetie. The";
    test_generate("That was the darkest day of the year.", "stories15M.bin", 0.7f, 100, 0.9f, expected);

const char* expected2="It was dark and cold around. The little girl was feeling scared. She looked around and saw a big, dark room. She wanted to go in, but she was too scared.\n\
Suddenly, she heard a noise. It was coming from the corner of the room. She slowly walked over and saw a big, black cat. It was meowing and seemed to be trying to get her attention.\n\
The little girl was still scared, but she was also curious. She";

    test_generate ("It was dark and cold around.", "stories110M.bin", 0.3f, 103, 0.6f, expected2);

    const char* expected3 = "There was a boy, who was three years old. He loved to play in the park. One day, he saw a fountain in the park and he wanted to play in it. He was very excited and he started running towards the fountain. But when he got close, he slipped and fell into the fountain. He was so embarrassed. \n\
The boy started crying and he felt very sad. He wanted to get out of the fountain, but he couldn't. He started to cry even louder. \n\
Suddenly, a kind old man came to the fountain and he helped the boy out of the fountain. He was very kind and he gave the boy a hug. The boy was very happy and he thanked the old man. \n\
The boy was very embarrassed, but he was also very happy. He learned that it was important to be careful when playing in";
    test_generate("There was a boy, who", "stories110M.bin", 0.7f, 200, 0.9f, expected3);

      const char* expected4 = "While you sleep, i`m destroying this world. She was so angry and frustrated that she couldn't stop. She was so angry that she started to cry.\n\
Suddenly, she heard a voice. It was coming from outside her window. It was a little bird.\n\
\"Why are you so angry,Anyie?\" asked the bird.\n\
\"I want to destroy this world,\" saidAnyie.\n\
The bird smiled. \"You can't";
    test_generate("While you sleep, i`m destroying this world.", "stories110M.bin", 0.2f, 100, 0.9f, expected4);

    const char* expected5 = "It was cold and it rained. The sky was grey and the trees were wet.\n\
Mommy and Daddy were busy in the kitchen. They were getting ready to make dinner. They put on their coats and hats.\n\
Then they heard a loud noise outside. It was thunder! It was so loud that it made them jump.\n\
Mommy and Daddy opened the window and saw a big, dark cloud. They were scared, so they decided to stay inside.\n\
Mommy and Daddy started to make dinner. They put some food on the table and started to eat. But then the thunder came again. It was so loud that it made the lights go out.\n\
Mommy and Daddy were scared and tried to hide. But it was too late. The thunder was too loud and it made the lights go out.\n\
The thunder was so strong that it made a big storm. It was so bad";
    test_generate("It was cold and it rained.", "stories110M.bin", 0.75f, 200, 0.8f, expected5);

    printf("ALL OK\n");

}
