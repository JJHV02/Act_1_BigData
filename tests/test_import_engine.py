def test_import_engine():
    import decision_engine as de
    assert isinstance(de.DEFAULT_CONFIG, dict)
