from homeassistant import config_entries
from homeassistant.helpers.selector import selector
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import urllib.parse
from .const import (
    DOMAIN,
    CONF_OPENAI_API_KEY,
    CONF_ANTHROPIC_API_KEY,
    CONF_GOOGLE_API_KEY,
    CONF_GROQ_API_KEY,
    CONF_LOCALAI_IP_ADDRESS,
    CONF_LOCALAI_PORT,
    CONF_LOCALAI_HTTPS,
    CONF_OLLAMA_IP_ADDRESS,
    CONF_OLLAMA_PORT,
    CONF_OLLAMA_HTTPS,
    CONF_CUSTOM_OPENAI_API_KEY,
    CONF_CUSTOM_OPENAI_ENDPOINT,
    VERSION_ANTHROPIC,
    CONF_RETENTION_TIME,
    CONF_OPENROUTER_API_KEY,
    DEFAULT_MODEL_OPENROUTER,
    OPENROUTER_VISION_MODELS,
    ENDPOINT_OPENROUTER,
)
import voluptuous as vol
import logging

_LOGGER = logging.getLogger(__name__)


class Validator:
    def __init__(self, hass, user_input):
        self.hass = hass
        self.user_input = user_input

    async def _validate_api_key(self, api_key):
        if not api_key or api_key == "":
            _LOGGER.error("You need to provide a valid API key.")
            raise ServiceValidationError("empty_api_key")
        elif self.user_input["provider"] == "OpenAI":
            header = {'Content-type': 'application/json',
                      'Authorization': 'Bearer ' + api_key}
            base_url = "api.openai.com"
            endpoint = "/v1/models"
            payload = {}
            method = "GET"
        elif self.user_input["provider"] == "Anthropic":
            header = {
                'x-api-key': api_key,
                'content-type': 'application/json',
                'anthropic-version': VERSION_ANTHROPIC
            }
            payload = {
                "model": "claude-3-haiku-20240307",
                "messages": [
                    {"role": "user", "content": "Hello, world"}
                ],
                "max_tokens": 50,
                "temperature": 0.5
            }
            base_url = "api.anthropic.com"
            endpoint = "/v1/messages"
            method = "POST"
        elif self.user_input["provider"] == "Google":
            header = {"content-type": "application/json"}
            base_url = "generativelanguage.googleapis.com"
            endpoint = f"/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": "Hello"}
                    ]}
                ]
            }
            method = "POST"
        elif self.user_input["provider"] == "Groq":
            header = {
                'Authorization': 'Bearer ' + api_key,
                'Content-Type': 'application/json'
            }
            base_url = "api.groq.com"
            endpoint = "/openai/v1/chat/completions"
            payload = {"messages": [
                {"role": "user", "content": "Hello"}], "model": "gemma-7b-it"}
            method = "POST"
        elif self.user_input["provider"] == "OpenRouter":
            header = {
                'Authorization': 'Bearer ' + api_key,
                'Content-Type': 'application/json'
            }
            base_url = "openrouter.ai"
            endpoint = "/api/v1/models"  
            payload = {}
            method = "GET"

        return await self._handshake(base_url=base_url, endpoint=endpoint, protocol="https", header=header, payload=payload, expected_status=200, method=method)

    def _validate_provider(self):
        """Validate provider configuration."""
        provider = self.user_input.get("provider")
        if not provider:
            _LOGGER.error("No provider specified")
            raise ServiceValidationError("no_provider")

    async def _handshake(self, base_url, endpoint, protocol="http", port="", header={}, payload={}, expected_status=200, method="GET"):
        _LOGGER.debug(
            f"Connecting to {protocol}://{base_url}{port}{endpoint}")
        session = async_get_clientsession(self.hass)
        url = f'{protocol}://{base_url}{port}{endpoint}'
        try:
            if method == "GET":
                response = await session.get(url, headers=header)
            elif method == "POST":
                response = await session.post(url, headers=header, json=payload)
            if response.status == expected_status:
                return True
            else:
                _LOGGER.error(
                    f"Handshake failed with status: {response.status}")
                return False
        except Exception as e:
            _LOGGER.error(f"Could not connect to {url}: {e}")
            return False

    async def localai(self):
        self._validate_provider()
        if not self.user_input[CONF_LOCALAI_IP_ADDRESS]:
            raise ServiceValidationError("empty_ip_address")
        if not self.user_input[CONF_LOCALAI_PORT]:
            raise ServiceValidationError("empty_port")
        protocol = "https" if self.user_input[CONF_LOCALAI_HTTPS] else "http"
        if not await self._handshake(base_url=self.user_input[CONF_LOCALAI_IP_ADDRESS], port=":"+str(self.user_input[CONF_LOCALAI_PORT]), protocol=protocol, endpoint="/readyz"):
            _LOGGER.error("Could not connect to LocalAI server.")
            raise ServiceValidationError("handshake_failed")

    async def ollama(self):
        self._validate_provider()
        if not self.user_input[CONF_OLLAMA_IP_ADDRESS]:
            raise ServiceValidationError("empty_ip_address")
        if not self.user_input[CONF_OLLAMA_PORT]:
            raise ServiceValidationError("empty_port")
        protocol = "https" if self.user_input[CONF_OLLAMA_HTTPS] else "http"
        if not await self._handshake(base_url=self.user_input[CONF_OLLAMA_IP_ADDRESS], port=":"+str(self.user_input[CONF_OLLAMA_PORT]), protocol=protocol, endpoint="/api/tags"):
            _LOGGER.error("Could not connect to Ollama server.")
            raise ServiceValidationError("handshake_failed")

    async def openai(self):
        self._validate_provider()
        if not await self._validate_api_key(self.user_input[CONF_OPENAI_API_KEY]):
            _LOGGER.error("Could not connect to OpenAI server.")
            raise ServiceValidationError("handshake_failed")

    async def custom_openai(self):
        self._validate_provider()
        try:
            url = urllib.parse.urlparse(
                self.user_input[CONF_CUSTOM_OPENAI_ENDPOINT])
            protocol = url.scheme
            base_url = url.hostname
            path = url.path if url.path else ""
            port = ":" + str(url.port) if url.port else ""

            endpoint = path + "/v1/models"
            header = {'Content-type': 'application/json',
                      'Authorization': 'Bearer ' + self.user_input[CONF_CUSTOM_OPENAI_API_KEY]} if CONF_CUSTOM_OPENAI_API_KEY in self.user_input else {}
        except Exception as e:
            _LOGGER.error(f"Could not parse endpoint: {e}")
            raise ServiceValidationError("endpoint_parse_failed")

        _LOGGER.debug(
            f"Connecting to: [protocol: {protocol}, base_url: {base_url}, port: {port}, endpoint: {endpoint}]")

        if not await self._handshake(base_url=base_url, port=port, protocol=protocol, endpoint=endpoint, header=header):
            _LOGGER.error("Could not connect to Custom OpenAI server.")
            raise ServiceValidationError("handshake_failed")

    async def anthropic(self):
        self._validate_provider()
        if not await self._validate_api_key(self.user_input[CONF_ANTHROPIC_API_KEY]):
            _LOGGER.error("Could not connect to Anthropic server.")
            raise ServiceValidationError("handshake_failed")

    async def google(self):
        self._validate_provider()
        if not await self._validate_api_key(self.user_input[CONF_GOOGLE_API_KEY]):
            _LOGGER.error("Could not connect to Google server.")
            raise ServiceValidationError("handshake_failed")

    async def groq(self):
        self._validate_provider()
        if not await self._validate_api_key(self.user_input[CONF_GROQ_API_KEY]):
            _LOGGER.error("Could not connect to Groq server.")
            raise ServiceValidationError("handshake_failed")

    async def openrouter(self):
        """Validate OpenRouter configuration."""
        self._validate_provider()
        api_key = self.user_input[CONF_OPENROUTER_API_KEY]
        
        if not api_key:
            _LOGGER.error("You need to provide a valid API key.")
            raise ServiceValidationError("empty_api_key")

        # Validate model if provided
        model = self.user_input.get("default_model")
        if model and model not in OPENROUTER_VISION_MODELS and not model.count("/") == 1:
            _LOGGER.error(f"Invalid model format: {model}. Must be in format 'provider/model'")
            raise ServiceValidationError("invalid_model_format")

        header = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/valentinfrlch/ha-llmvision',
            'X-Title': 'Home Assistant LLM Vision'
        }
        base_url = "openrouter.ai"
        endpoint = "/api/v1/models"
        payload = {}
        method = "GET"

        if not await self._handshake(base_url=base_url, endpoint=endpoint, protocol="https", header=header, payload=payload, expected_status=200, method=method):
            _LOGGER.error("Could not connect to OpenRouter server.")
            raise ServiceValidationError("handshake_failed")

    async def semantic_index(self) -> bool:
        """Validate Event Calendar configuration."""
        retention_time = self.user_input.get(CONF_RETENTION_TIME)
        
        # Validate retention time
        if retention_time < 0:
            raise ServiceValidationError("Retention time cannot be negative")
            
        return True

    def get_configured_providers(self):
        providers = []
        try:
            if self.hass.data[DOMAIN] is None:
                return providers
        except KeyError:
            return providers
        if CONF_OPENAI_API_KEY in self.hass.data[DOMAIN]:
            providers.append("OpenAI")
        if CONF_ANTHROPIC_API_KEY in self.hass.data[DOMAIN]:
            providers.append("Anthropic")
        if CONF_GOOGLE_API_KEY in self.hass.data[DOMAIN]:
            providers.append("Google")
        if CONF_LOCALAI_IP_ADDRESS in self.hass.data[DOMAIN] and CONF_LOCALAI_PORT in self.hass.data[DOMAIN]:
            providers.append("LocalAI")
        if CONF_OLLAMA_IP_ADDRESS in self.hass.data[DOMAIN] and CONF_OLLAMA_PORT in self.hass.data[DOMAIN]:
            providers.append("Ollama")
        if CONF_CUSTOM_OPENAI_ENDPOINT in self.hass.data[DOMAIN]:
            providers.append("Custom OpenAI")
        if CONF_GROQ_API_KEY in self.hass.data[DOMAIN]:
            providers.append("Groq")
        if CONF_OPENROUTER_API_KEY in self.hass.data[DOMAIN]:
            providers.append("OpenRouter")
        return providers


class llmvisionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):

    VERSION = 2

    async def async_step_user(self, user_input=None):
        """Handle a flow initialized by the user."""
        return self.async_show_menu(
            step_id="user",
            menu_options=["llm_providers", "non_llm_options"]
        )

    async def async_step_llm_providers(self, user_input=None):
        """Handle LLM providers setup."""
        data_schema = vol.Schema({
            vol.Required("provider", default="OpenAI"): selector({
                "select": {
                    "options": ["OpenAI", "Anthropic", "Google", "Groq", "OpenRouter", "LocalAI", "Ollama", "Custom OpenAI"],
                    "mode": "dropdown",
                    "sort": False,
                    "custom_value": False
                }
            }),
        })

        if user_input is not None:
            provider = user_input["provider"]
            return await self.handle_provider(provider)

        return self.async_show_form(
            step_id="llm_providers",
            data_schema=data_schema,
        )

    async def async_step_non_llm_options(self, user_input=None):
        """Handle non-LLM options setup."""
        data_schema = vol.Schema({
            vol.Required("provider", default="Event Calendar"): str,
            vol.Optional("retention_time", default=7): int
        })

        if user_input is not None:
            # Check if Event Calendar is already configured
            for entry in self.hass.config_entries.async_entries(DOMAIN):
                if entry.data.get("provider") == "Event Calendar":
                    return self.async_abort(reason="already_configured")
            
            # Add provider to user_input
            user_input["provider"] = "Event Calendar"
            
            validator = Validator(self.hass, user_input)
            await validator.semantic_index()
            
            return self.async_create_entry(
                title="LLM Vision Events",
                data=user_input
            )

        return self.async_show_form(
            step_id="non_llm_options",
            data_schema=data_schema,
        )

    async def handle_provider(self, provider):
        provider_steps = {
            "OpenAI": self.async_step_openai,
            "Anthropic": self.async_step_anthropic,
            "Google": self.async_step_google,
            "Groq": self.async_step_groq,
            "OpenRouter": self.async_step_openrouter,
            "LocalAI": self.async_step_localai,
            "Ollama": self.async_step_ollama,
            "Custom OpenAI": self.async_step_custom_openai,
        }

        step_method = provider_steps.get(provider)
        if step_method:
            return await step_method()
        else:
            _LOGGER.error(f"Unknown provider: {provider}")
            return self.async_abort(reason="unknown_provider")

    async def async_step_localai(self, user_input=None):
        data_schema = vol.Schema({
            vol.Required(CONF_LOCALAI_IP_ADDRESS): str,
            vol.Required(CONF_LOCALAI_PORT, default=8080): int,
            vol.Required(CONF_LOCALAI_HTTPS, default=False): bool,
        })

        if user_input is not None:
            # save provider to user_input
            user_input["provider"] = "LocalAI"
            validator = Validator(self.hass, user_input)
            try:
                await validator.localai()
                # add the mode to user_input
                return self.async_create_entry(title=f"LocalAI ({user_input[CONF_LOCALAI_IP_ADDRESS]})", data=user_input)
            except ServiceValidationError as e:
                _LOGGER.error(f"Validation failed: {e}")
                return self.async_show_form(
                    step_id="localai",
                    data_schema=data_schema,
                    errors={"base": "handshake_failed"}
                )

        return self.async_show_form(
            step_id="localai",
            data_schema=data_schema
        )

    async def async_step_ollama(self, user_input=None):
        data_schema = vol.Schema({
            vol.Required(CONF_OLLAMA_IP_ADDRESS): str,
            vol.Required(CONF_OLLAMA_PORT, default=11434): int,
            vol.Required(CONF_OLLAMA_HTTPS, default=False): bool,
        })

        if user_input is not None:
            # save provider to user_input
            user_input["provider"] = "Ollama"
            validator = Validator(self.hass, user_input)
            try:
                await validator.ollama()
                # add the mode to user_input
                return self.async_create_entry(title=f"Ollama ({user_input[CONF_OLLAMA_IP_ADDRESS]})", data=user_input)
            except ServiceValidationError as e:
                _LOGGER.error(f"Validation failed: {e}")
                return self.async_show_form(
                    step_id="ollama",
                    data_schema=data_schema,
                    errors={"base": "handshake_failed"}
                )

        return self.async_show_form(
            step_id="ollama",
            data_schema=data_schema,
        )

    async def async_step_openai(self, user_input=None):
        data_schema = vol.Schema({
            vol.Required(CONF_OPENAI_API_KEY): str,
        })

        if user_input is not None:
            # save provider to user_input
            user_input["provider"] = "OpenAI"
            validator = Validator(self.hass, user_input)
            try:
                await validator.openai()
                # add the mode to user_input
                user_input["provider"] = "OpenAI"
                return self.async_create_entry(title="OpenAI", data=user_input)
            except ServiceValidationError as e:
                _LOGGER.error(f"Validation failed: {e}")
                return self.async_show_form(
                    step_id="openai",
                    data_schema=data_schema,
                    errors={"base": "handshake_failed"}
                )

        return self.async_show_form(
            step_id="openai",
            data_schema=data_schema,
        )

    async def async_step_anthropic(self, user_input=None):
        data_schema = vol.Schema({
            vol.Required(CONF_ANTHROPIC_API_KEY): str,
        })

        if user_input is not None:
            # save provider to user_input
            user_input["provider"] = "Anthropic"
            validator = Validator(self.hass, user_input)
            try:
                await validator.anthropic()
                # add the mode to user_input
                user_input["provider"] = "Anthropic"
                return self.async_create_entry(title="Anthropic Claude", data=user_input)
            except ServiceValidationError as e:
                _LOGGER.error(f"Validation failed: {e}")
                return self.async_show_form(
                    step_id="anthropic",
                    data_schema=data_schema,
                    errors={"base": "empty_api_key"}
                )

        return self.async_show_form(
            step_id="anthropic",
            data_schema=data_schema,
        )

    async def async_step_google(self, user_input=None):
        data_schema = vol.Schema({
            vol.Required(CONF_GOOGLE_API_KEY): str,
        })

        if user_input is not None:
            # save provider to user_input
            user_input["provider"] = "Google"
            validator = Validator(self.hass, user_input)
            try:
                await validator.google()
                # add the mode to user_input
                user_input["provider"] = "Google"
                return self.async_create_entry(title="Google Gemini", data=user_input)
            except ServiceValidationError as e:
                _LOGGER.error(f"Validation failed: {e}")
                return self.async_show_form(
                    step_id="google",
                    data_schema=data_schema,
                    errors={"base": "empty_api_key"}
                )

        return self.async_show_form(
            step_id="google",
            data_schema=data_schema,
        )

    async def async_step_groq(self, user_input=None):
        data_schema = vol.Schema({
            vol.Required(CONF_GROQ_API_KEY): str,
        })

        if user_input is not None:
            # save provider to user_input
            user_input["provider"] = "Groq"
            validator = Validator(self.hass, user_input)
            try:
                await validator.groq()
                # add the mode to user_input
                user_input["provider"] = "Groq"
                return self.async_create_entry(title="Groq", data=user_input)
            except ServiceValidationError as e:
                _LOGGER.error(f"Validation failed: {e}")
                return self.async_show_form(
                    step_id="groq",
                    data_schema=data_schema,
                    errors={"base": "handshake_failed"}
                )

        return self.async_show_form(
            step_id="groq",
            data_schema=data_schema,
        )

    async def async_step_openrouter(self, user_input=None):
        """Handle OpenRouter configuration."""
        errors = {}
        data_schema = vol.Schema({
            vol.Required(CONF_OPENROUTER_API_KEY): str,
            vol.Optional("default_model", default=DEFAULT_MODEL_OPENROUTER): selector({
                "select": {
                    "options": OPENROUTER_VISION_MODELS,
                    "mode": "dropdown",
                    "sort": False,
                    "custom_value": True
                }
            }),
        })

        if user_input is not None:
            try:
                # Add provider to user_input before validation
                user_input["provider"] = "OpenRouter"
                validator = Validator(self.hass, user_input)
                await validator.openrouter()
                
                # Store the API key and model info together
                return self.async_create_entry(
                    title=f"OpenRouter ({user_input.get('default_model', DEFAULT_MODEL_OPENROUTER)})",
                    data=user_input
                )
            except ServiceValidationError as error:
                errors["base"] = error.args[0]

        return self.async_show_form(
            step_id="openrouter",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_custom_openai(self, user_input=None):
        data_schema = vol.Schema({
            vol.Required(CONF_CUSTOM_OPENAI_ENDPOINT): str,
            vol.Optional(CONF_CUSTOM_OPENAI_API_KEY): str,
        })

        if user_input is not None:
            # save provider to user_input
            user_input["provider"] = "Custom OpenAI"
            validator = Validator(self.hass, user_input)
            try:
                await validator.custom_openai()
                # add the mode to user_input
                user_input["provider"] = "Custom OpenAI"
                return self.async_create_entry(title="Custom OpenAI compatible Provider", data=user_input)
            except ServiceValidationError as e:
                _LOGGER.error(f"Validation failed: {e}")
                return self.async_show_form(
                    step_id="custom_openai",
                    data_schema=data_schema,
                    errors={"base": "handshake_failed"}
                )

        return self.async_show_form(
            step_id="custom_openai",
            data_schema=data_schema,
        )
