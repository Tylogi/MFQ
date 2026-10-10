from mfq.server.api import create_contract_app
from mfq.server.api.openapi import build_openapi_schema


def test_application_page_has_no_management_api():
    assert not any(path.startswith('/api/v1/applications') for path in build_openapi_schema()['paths'])
    assert not hasattr(create_contract_app().state, 'applications')
