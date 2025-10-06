#[derive(Debug)]
pub struct EvalReport {
    pub total: usize,
    pub passed: usize,
    pub cases: Vec<CaseReport>,
}

#[derive(Debug)]
pub struct CaseReport {
    pub name: String,
    pub pass: bool,
    pub failures: Vec<String>,
}