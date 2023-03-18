use llama_rs::{InferenceParameters, OutputToken};
use std::{io::Write, path::PathBuf, sync::Mutex};

fn main() {
    let model_path = std::env::args().nth(1).expect("model path");
    let model_path: PathBuf = model_path.into();

    let (model, vocab) =
        llama_rs::Model::load(model_path, 512, |_| {}).expect("could not load model");
    let mut session = model.start_session(64);

    let mut rng = rand::thread_rng();
    let inference_params = InferenceParameters::default();

    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        let prompt = format!(
            r#"---

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

{}

### Response:

"#,
            input
        );

        let i_love_spamming_mutexes = Mutex::new(String::new());

        session
            .inference_with_prompt(&model, &vocab, &inference_params, &prompt, &mut rng, |t| {
                let mut prompt_so_far = i_love_spamming_mutexes.lock().unwrap();

                match t {
                    OutputToken::Token(t) => {
                        let prev_starts_with = prompt_so_far.starts_with(&prompt);
                        prompt_so_far.push_str(t);
                        let now_starts_with = prompt_so_far.starts_with(&prompt);

                        if prev_starts_with && now_starts_with {
                            print!("{t}");
                            std::io::stdout().flush().unwrap();
                        }
                    }
                    OutputToken::EndOfText => {
                        println!();
                        std::io::stdout().flush().unwrap();
                    }
                }
            })
            .expect("could not run inference");
    }
}
