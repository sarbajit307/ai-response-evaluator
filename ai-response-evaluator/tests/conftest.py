import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup test environment variables BEFORE importing config/app modules
os.environ["DATABASE_URL"] = "sqlite:///./test_evaluator.db"
os.environ["LLM_PROVIDER"] = "mock" # Force mock mode for fast unit testing without API keys

from backend.app.database.session import Base, get_db
from backend.app.main import app

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_evaluator.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def init_test_db():
    """Initializes tables for test runs and cleans up afterwards."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    # Remove test DB file
    if os.path.exists("./test_evaluator.db"):
        try:
            os.remove("./test_evaluator.db")
        except PermissionError:
            pass

@pytest.fixture(name="db_session")
def fixture_db_session():
    """Provides isolated DB sessions for individual test assertions."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(name="client")
def fixture_client(db_session):
    """Provides a TestClient with overridden database dependencies."""
    from fastapi.testclient import TestClient
    
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
            
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
