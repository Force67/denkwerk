use denkwerk::{
    ChatHistory,
    ConciseSummarizer,
    FixedWindowCompressor,
};

fn main() {
    let mut history = ChatHistory::new();
    let transcript = [
        ("user", "Hey there! I'm planning a weekend trip."),
        ("assistant", "Great! Where are you thinking of going?"),
        ("user", "Somewhere warm, maybe the coast."),
        ("assistant", "How about checking out Santa Barbara?"),
        ("user", "That sounds promising. What should I pack?"),
        ("assistant", "Light layers, sunscreen, and a hat should do."),
        ("user", "Any good local spots for seafood?"),
        ("assistant", "Try the harbor restaurants for fresh catch."),
    ];

    for (role, text) in transcript {
        match role {
            "user" => history.push_user(text),
            _ => history.push_assistant(text),
        }
    }

    println!("Before compression: {} messages", history.len());

    let mut compressor = FixedWindowCompressor::new(6, ConciseSummarizer::new(160));
    if history.compress(&mut compressor) {
        println!("History compressed to {} messages", history.len());
    }

    for message in history.iter() {
        let role = match &message.role {
            denkwerk::MessageRole::User => "User",
            denkwerk::MessageRole::Assistant => "Assistant",
            denkwerk::MessageRole::System => "System",
            denkwerk::MessageRole::Tool => "Tool",
        };
        let text = message.text().unwrap_or("");
        println!("[{role}] {text}");
    }
}
