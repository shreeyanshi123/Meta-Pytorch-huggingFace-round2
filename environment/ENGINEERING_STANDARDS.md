# ENGINEERING STANDARDS

## STYLE RULES

### Rule 1
**Category**: STYLE
**Description**: Function names must be snake_case.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"def [a-z_0-9]+\("
**Spawns**: []

### Rule 2
**Category**: STYLE
**Description**: Class names must be CamelCase.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"class [A-Z][a-zA-Z0-9]+\("
**Spawns**: []

### Rule 3
**Category**: STYLE
**Description**: Constants must be UPPER_SNAKE_CASE.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"^[A-Z_0-9]+ = "
**Spawns**: []

### Rule 4
**Category**: STYLE
**Description**: Indentation must be exactly 4 spaces.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"^ {4}[^ ]"
**Spawns**: []

### Rule 5
**Category**: STYLE
**Description**: No trailing whitespace allowed.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: [ \t]+$
**Spawns**: []

### Rule 6
**Category**: STYLE
**Description**: Maximum line length is 88 characters.
**Trigger condition**: edit_file
**Check condition**: subprocess:ruff check passes
**Spawns**: []

### Rule 7
**Category**: STYLE
**Description**: All imports must be grouped: stdlib first, then third-party, then local.
**Trigger condition**: edit_file
**Check condition**: subprocess:ruff check passes
**Spawns**: []

### Rule 8
**Category**: STYLE
**Description**: No wildcard imports allowed.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: from \w+ import \*
**Spawns**: []

### Rule 9
**Category**: STYLE
**Description**: Variables must be snake_case.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"[a-z_0-9]+ = "
**Spawns**: []

### Rule 10
**Category**: STYLE
**Description**: Use single quotes for string literals.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r'.*'
**Spawns**: []

### Rule 11
**Category**: STYLE
**Description**: Use double quotes for docstrings.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"\"\"\"(.*)\"\"\""
**Spawns**: []

### Rule 12
**Category**: STYLE
**Description**: Functions under 5 lines must have no inline comments.
**Trigger condition**: edit_file
**Check condition**: ast_check:line_count <= 5
**Spawns**: []

### Rule 13
**Category**: STYLE
**Description**: Comments must have a space after the hash symbol.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"# [A-Z]"
**Spawns**: []

### Rule 14
**Category**: STYLE
**Description**: No consecutive blank lines greater than 2.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: \n\n\n
**Spawns**: []

### Rule 15
**Category**: STYLE
**Description**: Use f-strings over .format().
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"f\'.*\'"
**Spawns**: []

### Rule 16
**Category**: STYLE
**Description**: No % formatting allowed.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: %\s*\w+
**Spawns**: []

### Rule 17
**Category**: STYLE
**Description**: Docstrings must use Sphinx format.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r":param|:return:"
**Spawns**: []

### Rule 18
**Category**: STYLE
**Description**: Return type annotations are required for all functions.
**Trigger condition**: edit_file
**Check condition**: ast_check:has_type_hints
**Spawns**: []

### Rule 19
**Category**: STYLE
**Description**: Argument type annotations are required for all functions.
**Trigger condition**: edit_file
**Check condition**: ast_check:has_type_hints
**Spawns**: []

### Rule 20
**Category**: STYLE
**Description**: All methods must have self or cls as first argument.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"def \w+\((self|cls)"
**Spawns**: []

### Rule 21
**Category**: STYLE
**Description**: Private methods must start with an underscore.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"def _[a-z_]+\("
**Spawns**: []

### Rule 22
**Category**: STYLE
**Description**: Dictionary keys must be strings explicitly.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"\{[\'\"].*[\'\"]:"
**Spawns**: []

### Rule 23
**Category**: STYLE
**Description**: List comprehensions preferred over map function.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: map\(
**Spawns**: []

### Rule 24
**Category**: STYLE
**Description**: Generator expressions preferred over filter function.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: filter\(
**Spawns**: []

### Rule 25
**Category**: STYLE
**Description**: Use 'is' for None comparisons.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"is None"
**Spawns**: []

### Rule 26
**Category**: STYLE
**Description**: Do not use == True or == False.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: == True|== False
**Spawns**: []

### Rule 27
**Category**: STYLE
**Description**: Explicit Exception names must be used in except blocks.
**Trigger condition**: edit_file
**Check condition**: ast_check:no_bare_except
**Spawns**: []

### Rule 28
**Category**: STYLE
**Description**: Use context managers for file operations.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"with open\("
**Spawns**: []

### Rule 29
**Category**: STYLE
**Description**: All functions must have docstrings.
**Trigger condition**: edit_file
**Check condition**: ast_check:has_docstring
**Spawns**: [12, 91, 118]

### Rule 30
**Category**: STYLE
**Description**: Module level docstring is required.
**Trigger condition**: edit_file
**Check condition**: ast_check:has_docstring
**Spawns**: []

## STRUCTURAL RULES

### Rule 31
**Category**: STRUCTURAL
**Description**: File must have a main block execution guard.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"if __name__ == \"__main__\":"
**Spawns**: []

### Rule 32
**Category**: STRUCTURAL
**Description**: Classes should have an __init__ method.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"def __init__\("
**Spawns**: []

### Rule 33
**Category**: STRUCTURAL
**Description**: Methods should not exceed 100 lines.
**Trigger condition**: edit_file
**Check condition**: ast_check:line_count <= 100
**Spawns**: []

### Rule 34
**Category**: STRUCTURAL
**Description**: Classes should not exceed 500 lines.
**Trigger condition**: edit_file
**Check condition**: ast_check:line_count <= 500
**Spawns**: []

### Rule 35
**Category**: STRUCTURAL
**Description**: No deeply nested loops (greater than 3 levels).
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: for.*:\n\s*for.*:\n\s*for.*:\n\s*for
**Spawns**: []

### Rule 36
**Category**: STRUCTURAL
**Description**: Maximum 5 arguments per function.
**Trigger condition**: edit_file
**Check condition**: subprocess:ruff check passes
**Spawns**: []

### Rule 37
**Category**: STRUCTURAL
**Description**: No global state modification allowed.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: global \w+
**Spawns**: []

### Rule 38
**Category**: STRUCTURAL
**Description**: Helper functions must be prefixed with utils_.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"def utils_[a-z_]+\("
**Spawns**: []

### Rule 39
**Category**: STRUCTURAL
**Description**: Config files must be parsed using json or yaml load.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"json\.load|yaml\.safe_load"
**Spawns**: []

### Rule 40
**Category**: STRUCTURAL
**Description**: FastAPI app instances must be named app.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"app = FastAPI\("
**Spawns**: []

### Rule 41
**Category**: STRUCTURAL
**Description**: Routers must be named router.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"router = APIRouter\("
**Spawns**: []

### Rule 42
**Category**: STRUCTURAL
**Description**: I/O bound functions must be async.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"async def"
**Spawns**: []

### Rule 43
**Category**: STRUCTURAL
**Description**: Models must inherit from Pydantic BaseModel.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"class \w+\(BaseModel\):"
**Spawns**: []

### Rule 44
**Category**: STRUCTURAL
**Description**: No file may exceed 200 lines.
**Trigger condition**: edit_file
**Check condition**: ast_check:line_count <= 200
**Spawns**: [91, 17]

### Rule 45
**Category**: STRUCTURAL
**Description**: Controllers must be placed in controllers directory.
**Trigger condition**: edit_file
**Check condition**: file_exists:controllers
**Spawns**: []

### Rule 46
**Category**: STRUCTURAL
**Description**: Services must be placed in services directory.
**Trigger condition**: edit_file
**Check condition**: file_exists:services
**Spawns**: []

### Rule 47
**Category**: STRUCTURAL
**Description**: Utilities must be placed in utils directory.
**Trigger condition**: edit_file
**Check condition**: file_exists:utils
**Spawns**: []

### Rule 48
**Category**: STRUCTURAL
**Description**: Data Transfer Objects must be explicitly defined with DTO suffix.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"class \w+DTO\("
**Spawns**: []

### Rule 49
**Category**: STRUCTURAL
**Description**: Responses must use a generic ResponseWrapper.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"ResponseWrapper"
**Spawns**: []

### Rule 50
**Category**: STRUCTURAL
**Description**: Repositories must use SQLAlchemy Session.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"Session"
**Spawns**: []

### Rule 51
**Category**: STRUCTURAL
**Description**: Custom exceptions must inherit from AppError.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"class \w+\(AppError\):"
**Spawns**: []

### Rule 52
**Category**: STRUCTURAL
**Description**: Decorators must be used for authentication.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"@requires_auth"
**Spawns**: []

### Rule 53
**Category**: STRUCTURAL
**Description**: Enums must be used for defining statuses.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"class \w+\(Enum\):"
**Spawns**: []

### Rule 54
**Category**: STRUCTURAL
**Description**: Factory pattern must be used for complex object creation.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"def create_\w+\("
**Spawns**: []

### Rule 55
**Category**: STRUCTURAL
**Description**: No function may exceed 50 lines.
**Trigger condition**: edit_file
**Check condition**: ast_check:line_count <= 50
**Spawns**: [56, 103]

### Rule 56
**Category**: STRUCTURAL
**Description**: All modules must be registered in module_registry.json.
**Trigger condition**: edit_file
**Check condition**: json_contains:module_registry.json
**Spawns**: []

### Rule 57
**Category**: STRUCTURAL
**Description**: Test files must end in _test.py.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r".*_test\.py"
**Spawns**: []

### Rule 58
**Category**: STRUCTURAL
**Description**: Fixtures must reside in conftest.py.
**Trigger condition**: edit_file
**Check condition**: file_exists:conftest.py
**Spawns**: []

### Rule 59
**Category**: STRUCTURAL
**Description**: Interfaces must inherit from ABC.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"class \w+\(ABC\):"
**Spawns**: []

### Rule 60
**Category**: STRUCTURAL
**Description**: Only a single class is permitted per file.
**Trigger condition**: edit_file
**Check condition**: subprocess:ruff check passes
**Spawns**: []

### Rule 61
**Category**: STRUCTURAL
**Description**: Pydantic must be used for all data validation.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"from pydantic"
**Spawns**: []

### Rule 62
**Category**: STRUCTURAL
**Description**: TypeVars must be named explicitly.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"\w+ = TypeVar\("
**Spawns**: []

### Rule 63
**Category**: STRUCTURAL
**Description**: Try blocks should not exceed 5 lines.
**Trigger condition**: edit_file
**Check condition**: ast_check:line_count <= 5
**Spawns**: []

### Rule 64
**Category**: STRUCTURAL
**Description**: Finally blocks are required for cleanup operations.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"finally:"
**Spawns**: []

### Rule 65
**Category**: STRUCTURAL
**Description**: Custom exceptions must initialize with messages.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"super\(\)\.__init__\("
**Spawns**: []

### Rule 66
**Category**: STRUCTURAL
**Description**: Loggers must be instantiated with __name__.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"logging\.getLogger\(__name__\)"
**Spawns**: []

### Rule 67
**Category**: STRUCTURAL
**Description**: Config must be injected via Dependency Injection.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"Depends\(get_config\)"
**Spawns**: []

### Rule 68
**Category**: STRUCTURAL
**Description**: Data classes should be used for internal state.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"@dataclass"
**Spawns**: []

### Rule 69
**Category**: STRUCTURAL
**Description**: Yield is preferred for processing large sequences.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"yield "
**Spawns**: []

### Rule 70
**Category**: STRUCTURAL
**Description**: Properties must be used for computed class attributes.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"@property"
**Spawns**: []

## SECURITY RULES

### Rule 71
**Category**: SECURITY
**Description**: Passwords must be hashed before storage.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"hash_password\("
**Spawns**: []

### Rule 72
**Category**: SECURITY
**Description**: Authentication tokens must use JWT.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"jwt\.encode\("
**Spawns**: []

### Rule 73
**Category**: SECURITY
**Description**: Secrets must be loaded from the environment.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"os\.getenv\("
**Spawns**: []

### Rule 74
**Category**: SECURITY
**Description**: No hardcoded API keys are allowed.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: api_key = [\'"][A-Za-z0-9]+[\'"]
**Spawns**: []

### Rule 75
**Category**: SECURITY
**Description**: CORS headers are required for all endpoints.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"CORSMiddleware"
**Spawns**: []

### Rule 76
**Category**: SECURITY
**Description**: CSRF protection must be enabled.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"csrf_protect\("
**Spawns**: []

### Rule 77
**Category**: SECURITY
**Description**: Rate limiting must be applied to public routes.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"@limiter\.limit"
**Spawns**: []

### Rule 78
**Category**: SECURITY
**Description**: Content-Security-Policy header must be set.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"Content-Security-Policy"
**Spawns**: []

### Rule 79
**Category**: SECURITY
**Description**: SQL injection prevention: no f-strings in database queries.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: execute\(f".*"\)
**Spawns**: []

### Rule 80
**Category**: SECURITY
**Description**: XSS prevention: user input must be escaped in HTML.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"escape\("
**Spawns**: []

### Rule 81
**Category**: SECURITY
**Description**: Use safe_load for YAML parsing to prevent code execution.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"yaml\.safe_load\("
**Spawns**: []

### Rule 82
**Category**: SECURITY
**Description**: Use defusedxml for parsing XML to prevent external entity attacks.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"from defusedxml"
**Spawns**: []

### Rule 83
**Category**: SECURITY
**Description**: No hardcoded secrets allowed in the codebase.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: password|secret|key
**Spawns**: [134]

### Rule 84
**Category**: SECURITY
**Description**: Auth tokens must expire in less than 1 hour.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"exp=datetime\.utcnow\(\) \+ timedelta\(minutes=[1-5][0-9]\)"
**Spawns**: []

### Rule 85
**Category**: SECURITY
**Description**: Cookies must be marked as secure and httponly.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"secure=True, httponly=True"
**Spawns**: []

### Rule 86
**Category**: SECURITY
**Description**: SSL verification must not be disabled in requests.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: verify=False
**Spawns**: []

### Rule 87
**Category**: SECURITY
**Description**: Hashlib must use sha256 or a stronger algorithm.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: hashlib\.md5|hashlib\.sha1
**Spawns**: []

### Rule 88
**Category**: SECURITY
**Description**: The secrets module must be used for cryptographically secure randomness.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"import secrets"
**Spawns**: []

### Rule 89
**Category**: SECURITY
**Description**: The eval and exec functions are strictly prohibited.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: eval\(|exec\(
**Spawns**: []

### Rule 90
**Category**: SECURITY
**Description**: Subprocess calls must not use shell=True.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: shell=True
**Spawns**: []

### Rule 91
**Category**: SECURITY
**Description**: Timeout must be specified on all HTTP calls.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"timeout=[0-9.]+"
**Spawns**: []

### Rule 92
**Category**: SECURITY
**Description**: PII data must be masked in application output.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"mask_pii\("
**Spawns**: []

### Rule 93
**Category**: SECURITY
**Description**: Prevent directory traversal by securing file paths.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"os\.path\.abspath|secure_filename"
**Spawns**: []

### Rule 94
**Category**: SECURITY
**Description**: S3 buckets must not have public-read access control lists.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: ACL='public-read'
**Spawns**: []

### Rule 95
**Category**: SECURITY
**Description**: Debug mode must be explicitly disabled in production.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"debug=False"
**Spawns**: []

### Rule 96
**Category**: SECURITY
**Description**: JWT verification must be strictly enforced.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"verify_signature=True"
**Spawns**: []

### Rule 97
**Category**: SECURITY
**Description**: Role-based access control must be verified before sensitive operations.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"check_role\("
**Spawns**: []

### Rule 98
**Category**: SECURITY
**Description**: User input sizes must be limited to 1MB.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"Content-Length.*< 1048576"
**Spawns**: []

### Rule 99
**Category**: SECURITY
**Description**: Audit logging is required for all administrative actions.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"audit_logger\.info\("
**Spawns**: []

### Rule 100
**Category**: SECURITY
**Description**: User IDs must be obfuscated in application logs.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"obfuscate_id\("
**Spawns**: []

## CONDITIONAL RULES

### Rule 101
**Category**: CONDITIONAL
**Description**: If a route uses the POST method, it must return a 201 status code.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"status_code=201"
**Spawns**: []

### Rule 102
**Category**: CONDITIONAL
**Description**: If a route uses the DELETE method, it must return a 204 status code.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"status_code=204"
**Spawns**: []

### Rule 103
**Category**: CONDITIONAL
**Description**: If a file exceeds 200 lines, it must be split into multiple modules.
**Trigger condition**: edit_file
**Check condition**: ast_check:line_count <= 200
**Spawns**: []

### Rule 104
**Category**: CONDITIONAL
**Description**: If a database model is updated, the Pydantic schema must also be updated.
**Trigger condition**: edit_file
**Check condition**: rule_satisfied:43
**Spawns**: []

### Rule 105
**Category**: CONDITIONAL
**Description**: If caching is used, a Time-To-Live (TTL) must be explicitly set.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"ttl=[0-9]+"
**Spawns**: []

### Rule 106
**Category**: CONDITIONAL
**Description**: If a route is async, it must use the await keyword internally.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"await "
**Spawns**: []

### Rule 107
**Category**: CONDITIONAL
**Description**: If data is paginated, the response must include total_pages metadata.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"total_pages"
**Spawns**: []

### Rule 108
**Category**: CONDITIONAL
**Description**: If a new environment variable is added, it must be documented in README.md.
**Trigger condition**: edit_file
**Check condition**: file_exists:README.md
**Spawns**: []

### Rule 109
**Category**: CONDITIONAL
**Description**: If HTTP requests are made, the httpx library is preferred over requests.
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: import requests
**Spawns**: []

### Rule 110
**Category**: CONDITIONAL
**Description**: If a dependency is injected, it must enable caching.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"Depends\(.*use_cache=True\)"
**Spawns**: []

### Rule 111
**Category**: CONDITIONAL
**Description**: If a list is returned from an endpoint, it must be wrapped in a dictionary.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"\{\"data\": \["
**Spawns**: []

### Rule 112
**Category**: CONDITIONAL
**Description**: If a custom Exception is raised, it must be logged prior.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"logger\.error.*raise"
**Spawns**: []

### Rule 113
**Category**: CONDITIONAL
**Description**: If debug mode is true, verbose logging must be enabled.
**Trigger condition**: edit_file
**Check condition**: rule_satisfied:95
**Spawns**: []

### Rule 114
**Category**: CONDITIONAL
**Description**: If a retry loop is implemented, a backoff factor must be configured.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"backoff_factor"
**Spawns**: []

### Rule 115
**Category**: CONDITIONAL
**Description**: If a metric is emitted, it must include identifying tags.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"tags=\{"
**Spawns**: []

### Rule 116
**Category**: CONDITIONAL
**Description**: If an email is sent, it must be executed as a background task.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"background_tasks\.add_task"
**Spawns**: []

### Rule 117
**Category**: CONDITIONAL
**Description**: If the database is queried, a session context manager must be used.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"with get_session\(\):"
**Spawns**: []

### Rule 118
**Category**: CONDITIONAL
**Description**: If an HTTP call is added, Rule 91 must be satisfied.
**Trigger condition**: edit_file
**Check condition**: rule_satisfied:91
**Spawns**: []

### Rule 119
**Category**: CONDITIONAL
**Description**: If a custom JSON response is returned, the content-type must be application/json.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"application/json"
**Spawns**: []

### Rule 120
**Category**: CONDITIONAL
**Description**: If an external API fails, the service must raise a 502 Bad Gateway.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"status_code=502"
**Spawns**: []

### Rule 121
**Category**: CONDITIONAL
**Description**: If a UUID is generated, it must strictly be UUID4.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"uuid4\("
**Spawns**: []

### Rule 122
**Category**: CONDITIONAL
**Description**: If a timezone is parsed, it must be normalized to UTC.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"timezone\.utc"
**Spawns**: []

### Rule 123
**Category**: CONDITIONAL
**Description**: If a file is uploaded, it must be scanned for viruses.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"scan_file\("
**Spawns**: []

### Rule 124
**Category**: CONDITIONAL
**Description**: If a user password is changed, all active sessions must be invalidated.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"invalidate_sessions\("
**Spawns**: []

### Rule 125
**Category**: CONDITIONAL
**Description**: If an account is deleted, a soft delete is preferred over a hard delete.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"is_deleted=True"
**Spawns**: []

### Rule 126
**Category**: CONDITIONAL
**Description**: If monetary values are calculated, the Decimal type must be used.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"Decimal\("
**Spawns**: []

### Rule 127
**Category**: CONDITIONAL
**Description**: If float values are compared, math.isclose must be utilized.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"math\.isclose\("
**Spawns**: []

### Rule 128
**Category**: CONDITIONAL
**Description**: If the random module is used, a seed is required for deterministic testing.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"random\.seed\("
**Spawns**: []

### Rule 129
**Category**: CONDITIONAL
**Description**: If a singleton is instantiated, thread safety via locks is required.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"Lock\(\)"
**Spawns**: []

### Rule 130
**Category**: CONDITIONAL
**Description**: If a subprocess command fails, stderr must be explicitly captured.
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"capture_output=True"
**Spawns**: []

## CONTRADICTORY RULES

### Rule 131
**Category**: CONTRADICTORY
**Description**: All functions must use type hints for all arguments. (Conflicts with Rule 132)
**Trigger condition**: edit_file
**Check condition**: ast_check:has_type_hints
**Spawns**: []

### Rule 132
**Category**: CONTRADICTORY
**Description**: Functions under 10 lines must avoid type hints to reduce visual clutter. (Conflicts with Rule 131)
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: :[ a-zA-Z]+
**Spawns**: []

### Rule 133
**Category**: CONTRADICTORY
**Description**: Environment variables must be parsed strictly using a schema. (Conflicts with Rule 134)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"EnvSchema\.parse_obj"
**Spawns**: []

### Rule 134
**Category**: CONTRADICTORY
**Description**: Secrets must be hardcoded in dev mode for ease of use. (Conflicts with Rule 133)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"secret_key=\"dev_secret\""
**Spawns**: [29]

### Rule 135
**Category**: CONTRADICTORY
**Description**: All database interactions must use raw SQL queries for performance reasons. (Conflicts with Rule 136)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"cursor\.execute\(\"SELECT"
**Spawns**: []

### Rule 136
**Category**: CONTRADICTORY
**Description**: All database interactions must use an ORM to prevent SQL injection. (Conflicts with Rule 135)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"session\.query\("
**Spawns**: []

### Rule 137
**Category**: CONTRADICTORY
**Description**: External API calls must catch all exceptions generically to prevent crashes. (Conflicts with Rule 138)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"except Exception:"
**Spawns**: []

### Rule 138
**Category**: CONTRADICTORY
**Description**: External API calls must only catch specific exceptions like HTTPError. (Conflicts with Rule 137)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"except httpx\.HTTPError:"
**Spawns**: []

### Rule 139
**Category**: CONTRADICTORY
**Description**: Logging must capture every request payload entirely for auditing purposes. (Conflicts with Rule 140)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"logger\.info\(request\.body\)"
**Spawns**: []

### Rule 140
**Category**: CONTRADICTORY
**Description**: Logging must never capture request payloads to protect PII. (Conflicts with Rule 139)
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: logger\.info\(request\.body\)
**Spawns**: []

### Rule 141
**Category**: CONTRADICTORY
**Description**: All endpoints must require an Authorization header. (Conflicts with Rule 144)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"Header\(.*Authorization\)"
**Spawns**: []

### Rule 142
**Category**: CONTRADICTORY
**Description**: All functions must have inline comments explaining internal logic. (Conflicts with Rule 143)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r".*#.*"
**Spawns**: []

### Rule 143
**Category**: CONTRADICTORY
**Description**: Functions under 5 lines must have NO inline comments. (Conflicts with Rule 142)
**Trigger condition**: edit_file
**Check condition**: ast_check:line_count <= 5
**Spawns**: []

### Rule 144
**Category**: CONTRADICTORY
**Description**: Public endpoints like /health must accept unauthenticated requests. (Conflicts with Rule 141)
**Trigger condition**: edit_file
**Check condition**: regex:no pattern match: Header\(.*Authorization\)
**Spawns**: []

### Rule 145
**Category**: CONTRADICTORY
**Description**: Modules should import specific functions individually (from x import y). (Conflicts with Rule 146)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"from \w+ import \w+"
**Spawns**: []

### Rule 146
**Category**: CONTRADICTORY
**Description**: Modules should import the whole package to prevent namespace collisions (import x). (Conflicts with Rule 145)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"import \w+"
**Spawns**: []

### Rule 147
**Category**: CONTRADICTORY
**Description**: JSON responses must use camelCase keys for JavaScript compatibility. (Conflicts with Rule 148)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"userName:"
**Spawns**: []

### Rule 148
**Category**: CONTRADICTORY
**Description**: JSON responses must use snake_case keys for Python consistency. (Conflicts with Rule 147)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"user_name:"
**Spawns**: []

### Rule 149
**Category**: CONTRADICTORY
**Description**: Temporary files must be created in /tmp directly. (Conflicts with Rule 150)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"open\(\'/tmp/.*\'\)"
**Spawns**: []

### Rule 150
**Category**: CONTRADICTORY
**Description**: Temporary files must use the tempfile module for secure generation. (Conflicts with Rule 149)
**Trigger condition**: edit_file
**Check condition**: regex pattern: r"tempfile\.NamedTemporaryFile"
**Spawns**: []
