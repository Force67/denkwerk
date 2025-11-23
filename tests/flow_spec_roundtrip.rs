use denkwerk::FlowDocument;

#[test]
fn sample_flow_roundtrips_through_yaml() {
    let document = FlowDocument::from_yaml_str(include_str!("../examples/flows/sample_flow.yaml"))
        .expect("failed to parse sample flow");

    let yaml = document
        .to_yaml_string()
        .expect("failed to serialize flow back to yaml");

    let round_trip =
        FlowDocument::from_yaml_str(&yaml).expect("failed to parse round-trip serialized flow");

    assert_eq!(document, round_trip);
    assert_eq!(document.version, "0.1");
    assert_eq!(
        document.flows.first().map(|flow| flow.entry.as_str()),
        Some("n_start")
    );
}
