(Files content cropped to 300k characters, download full ingest to see more)
================================================
FILE: README.md
================================================
# Google Gen AI SDK

[![PyPI version](https://img.shields.io/pypi/v/google-genai.svg)](https://pypi.org/project/google-genai/)

--------
**Documentation:** https://googleapis.github.io/python-genai/

-----

Google Gen AI Python SDK provides an interface for developers to integrate Google's generative models into their Python applications. It supports the [Gemini Developer API](https://ai.google.dev/gemini-api/docs) and [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview) APIs.

## Installation

```sh
pip install google-genai
```

## Imports

```python
from google import genai
from google.genai import types
```

## Create a client

Please run one of the following code blocks to create a client for
different services ([Gemini Developer API](https://ai.google.dev/gemini-api/docs) or [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview)).

```python
# Only run this block for Gemini Developer API
client = genai.Client(api_key='GEMINI_API_KEY')
```

```python
# Only run this block for Vertex AI API
client = genai.Client(
    vertexai=True, project='your-project-id', location='us-central1'
)
```

**(Optional) Using environment variables:**

You can create a client by configuring the necessary environment variables.
Configuration setup instructions depends on whether you're using the Gemini
Developer API or the Gemini API in Vertex AI.

**Gemini Developer API:** Set `GOOGLE_API_KEY` as shown below:

```bash
export GOOGLE_API_KEY='your-api-key'
```

**Gemini API on Vertex AI:** Set `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`
and `GOOGLE_CLOUD_LOCATION`, as shown below:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
```

```python
client = genai.Client()
```

### API Selection

By default, the SDK uses the beta API endpoints provided by Google to support
preview features in the APIs. The stable API endpoints can be selected by
setting the API version to `v1`.

To set the API version use `http_options`. For example, to set the API version
to `v1` for Vertex AI:

```python
client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1',
    http_options=types.HttpOptions(api_version='v1')
)
```

To set the API version to `v1alpha` for the Gemini Developer API:

```python
client = genai.Client(
    api_key='GEMINI_API_KEY',
    http_options=types.HttpOptions(api_version='v1alpha')
)
```

## Types

Parameter types can be specified as either dictionaries(`TypedDict`) or
[Pydantic Models](https://pydantic.readthedocs.io/en/stable/model.html).
Pydantic model types are available in the `types` module.

## Models

The `client.models` modules exposes model inferencing and model getters.

### Generate Content

#### with text content

```python
response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='Why is the sky blue?'
)
print(response.text)
```

#### with uploaded file (Gemini Developer API only)
download the file in console.

```sh
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

python code.

```python
file = client.files.upload(file='a11.txt')
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=['Could you summarize this file?', file]
)
print(response.text)
```

#### How to structure `contents` argument for `generate_content`
The SDK always converts the inputs to the `contents` argument into
`list[types.Content]`.
The following shows some common ways to provide your inputs.

##### Provide a `list[types.Content]`
This is the canonical way to provide contents, SDK will not do any conversion.

##### Provide a `types.Content` instance

```python
contents = types.Content(
  role='user',
  parts=[types.Part.from_text(text='Why is the sky blue?')]
)
```

SDK converts this to

```python
[
  types.Content(
    role='user',
    parts=[types.Part.from_text(text='Why is the sky blue?')]
  )
]
```

##### Provide a string

```python
contents='Why is the sky blue?'
```

The SDK will assume this is a text part, and it converts this into the following:

```python
[
  types.UserContent(
    parts=[
      types.Part.from_text(text='Why is the sky blue?')
    ]
  )
]
```

Where a `types.UserContent` is a subclass of `types.Content`, it sets the
`role` field to be `user`.

##### Provide a list of string

```python
contents=['Why is the sky blue?', 'Why is the cloud white?']
```

The SDK assumes these are 2 text parts, it converts this into a single content,
like the following:

```python
[
  types.UserContent(
    parts=[
      types.Part.from_text(text='Why is the sky blue?'),
      types.Part.from_text(text='Why is the cloud white?'),
    ]
  )
]
```

Where a `types.UserContent` is a subclass of `types.Content`, the
`role` field in `types.UserContent` is fixed to be `user`.

##### Provide a function call part

```python
contents = types.Part.from_function_call(
  name='get_weather_by_location',
  args={'location': 'Boston'}
)
```

The SDK converts a function call part to a content with a `model` role:

```python
[
  types.ModelContent(
    parts=[
      types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
      )
    ]
  )
]
```

Where a `types.ModelContent` is a subclass of `types.Content`, the
`role` field in `types.ModelContent` is fixed to be `model`.

##### Provide a list of function call parts

```python
contents = [
  types.Part.from_function_call(
    name='get_weather_by_location',
    args={'location': 'Boston'}
  ),
  types.Part.from_function_call(
    name='get_weather_by_location',
    args={'location': 'New York'}
  ),
]
```

The SDK converts a list of function call parts to the a content with a `model` role:

```python
[
  types.ModelContent(
    parts=[
      types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
      ),
      types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'New York'}
      )
    ]
  )
]
```

Where a `types.ModelContent` is a subclass of `types.Content`, the
`role` field in `types.ModelContent` is fixed to be `model`.

##### Provide a non function call part

```python
contents = types.Part.from_uri(
  file_uri: 'gs://generativeai-downloads/images/scones.jpg',
  mime_type: 'image/jpeg',
)
```

The SDK converts all non function call parts into a content with a `user` role.

```python
[
  types.UserContent(parts=[
    types.Part.from_uri(
     file_uri: 'gs://generativeai-downloads/images/scones.jpg',
      mime_type: 'image/jpeg',
    )
  ])
]
```

##### Provide a list of non function call parts

```python
contents = [
  types.Part.from_text('What is this image about?'),
  types.Part.from_uri(
    file_uri: 'gs://generativeai-downloads/images/scones.jpg',
    mime_type: 'image/jpeg',
  )
]
```

The SDK will convert the list of parts into a content with a `user` role

```python
[
  types.UserContent(
    parts=[
      types.Part.from_text('What is this image about?'),
      types.Part.from_uri(
        file_uri: 'gs://generativeai-downloads/images/scones.jpg',
        mime_type: 'image/jpeg',
      )
    ]
  )
]
```

##### Mix types in contents

You can also provide a list of `types.ContentUnion`. The SDK leaves items of
`types.Content` as is, it groups consecutive non function call parts into a
single `types.UserContent`, and it groups consecutive function call parts into
a single `types.ModelContent`.

If you put a list within a list, the inner list can only contain
`types.PartUnion` items. The SDK will convert the inner list into a single
`types.UserContent`.

### System Instructions and Other Configs

The output of the model can be influenced by several optional settings
available in generate_content's config parameter. For example, the
variability and length of the output can be influenced by the temperature
and max_output_tokens respectively.

```python
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say low',
        max_output_tokens=3,
        temperature=0.3,
    ),
)
print(response.text)
```

### Typed Config

All API methods support Pydantic types for parameters as well as
dictionaries. You can get the type from `google.genai.types`.

```python
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=types.Part.from_text(text='Why is the sky blue?'),
    config=types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        max_output_tokens=100,
        stop_sequences=['STOP!'],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)

print(response.text)
```

### List Base Models

To retrieve tuned models, see [list tuned models](#list-tuned-models).

```python
for model in client.models.list():
    print(model)
```

```python
pager = client.models.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

#### Async

```python
async for job in await client.aio.models.list():
    print(job)
```

```python
async_pager = await client.aio.models.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

### Safety Settings

```python
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH',
            )
        ]
    ),
)
print(response.text)
```

### Function Calling

#### Automatic Python function Support

You can pass a Python function directly and it will be automatically
called and responded by default.

```python
def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return 'sunny'


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(tools=[get_current_weather]),
)

print(response.text)
```
#### Disabling automatic function calling
If you pass in a python function as a tool directly, and do not want
automatic function calling, you can disable automatic function calling
as follows:

```python
response = client.models.generate_content(
  model='gemini-2.0-flash-001',
  contents='What is the weather like in Boston?',
  config=types.GenerateContentConfig(
    tools=[get_current_weather],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
      disable=True
    ),
  ),
)
```

With automatic function calling disabled, you will get a list of function call
parts in the response:

```python
function_calls: Optional[List[types.FunctionCall]] = response.function_calls
```

#### Manually declare and invoke a function for function calling

If you don't want to use the automatic function support, you can manually
declare the function and invoke it.

The following example shows how to declare a function and pass it as a tool.
Then you will receive a function call part in the response.

```python
function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'location': types.Schema(
                type='STRING',
                description='The city and state, e.g. San Francisco, CA',
            ),
        },
        required=['location'],
    ),
)

tool = types.Tool(function_declarations=[function])

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(tools=[tool]),
)

print(response.function_calls[0])
```

After you receive the function call part from the model, you can invoke the function
and get the function response. And then you can pass the function response to
the model.
The following example shows how to do it for a simple function invocation.

```python
user_prompt_content = types.Content(
    role='user',
    parts=[types.Part.from_text(text='What is the weather like in Boston?')],
)
function_call_part = response.function_calls[0]
function_call_content = response.candidates[0].content


try:
    function_result = get_current_weather(
        **function_call_part.function_call.args
    )
    function_response = {'result': function_result}
except (
    Exception
) as e:  # instead of raising the exception, you can let the model handle it
    function_response = {'error': str(e)}


function_response_part = types.Part.from_function_response(
    name=function_call_part.name,
    response=function_response,
)
function_response_content = types.Content(
    role='tool', parts=[function_response_part]
)

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=[
        user_prompt_content,
        function_call_content,
        function_response_content,
    ],
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)

print(response.text)
```

#### Function calling with `ANY` tools config mode

If you configure function calling mode to be `ANY`, then the model will always
return function call parts. If you also pass a python function as a tool, by
default the SDK will perform automatic function calling until the remote calls exceed the
maximum remote call for automatic function calling (default to 10 times).

If you'd like to disable automatic function calling in `ANY` mode:

```python
def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

If you'd like to set `x` number of automatic function call turns, you can
configure the maximum remote calls to be `x + 1`.
Assuming you prefer `1` turn for automatic function calling.

```python
def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=2
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```
### JSON Response Schema

However you define your schema, don't duplicate it in your input prompt,
including by giving examples of expected JSON output. If you do, the generated
output might be lower in quality.

#### Pydantic Model Schema support

Schemas can be provided as Pydantic Models.

```python
from pydantic import BaseModel


class CountryInfo(BaseModel):
    name: str
    population: int
    capital: str
    continent: str
    gdp: int
    official_language: str
    total_area_sq_mi: int


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=CountryInfo,
    ),
)
print(response.text)
```

```python
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema={
            'required': [
                'name',
                'population',
                'capital',
                'continent',
                'gdp',
                'official_language',
                'total_area_sq_mi',
            ],
            'properties': {
                'name': {'type': 'STRING'},
                'population': {'type': 'INTEGER'},
                'capital': {'type': 'STRING'},
                'continent': {'type': 'STRING'},
                'gdp': {'type': 'INTEGER'},
                'official_language': {'type': 'STRING'},
                'total_area_sq_mi': {'type': 'INTEGER'},
            },
            'type': 'OBJECT',
        },
    ),
)
print(response.text)
```

### Enum Response Schema

#### Text Response

You can set response_mime_type to 'text/x.enum' to return one of those enum
values as the response.

```python
class InstrumentEnum(Enum):
  PERCUSSION = 'Percussion'
  STRING = 'String'
  WOODWIND = 'Woodwind'
  BRASS = 'Brass'
  KEYBOARD = 'Keyboard'

response = client.models.generate_content(
      model='gemini-2.0-flash-001',
      contents='What instrument plays multiple notes at once?',
      config={
          'response_mime_type': 'text/x.enum',
          'response_schema': InstrumentEnum,
      },
  )
print(response.text)
```

#### JSON Response

You can also set response_mime_type to 'application/json', the response will be identical but in quotes.

```python
from enum import Enum

class InstrumentEnum(Enum):
  PERCUSSION = 'Percussion'
  STRING = 'String'
  WOODWIND = 'Woodwind'
  BRASS = 'Brass'
  KEYBOARD = 'Keyboard'

response = client.models.generate_content(
      model='gemini-2.0-flash-001',
      contents='What instrument plays multiple notes at once?',
      config={
          'response_mime_type': 'application/json',
          'response_schema': InstrumentEnum,
      },
  )
print(response.text)
```

### Streaming

#### Streaming for text content

```python
for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

#### Streaming for image content

If your image is stored in [Google Cloud Storage](https://cloud.google.com/storage),
you can use the `from_uri` class method to create a `Part` object.

```python
for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_uri(
            file_uri='gs://generativeai-downloads/images/scones.jpg',
            mime_type='image/jpeg',
        ),
    ],
):
    print(chunk.text, end='')
```

If your image is stored in your local file system, you can read it in as bytes
data and use the `from_bytes` class method to create a `Part` object.

```python
YOUR_IMAGE_PATH = 'your_image_path'
YOUR_IMAGE_MIME_TYPE = 'your_image_mime_type'
with open(YOUR_IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),
    ],
):
    print(chunk.text, end='')
```

### Async

`client.aio` exposes all the analogous [`async` methods](https://docs.python.org/3/library/asyncio.html)
that are available on `client`

For example, `client.aio.models.generate_content` is the `async` version
of `client.models.generate_content`

```python
response = await client.aio.models.generate_content(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
)

print(response.text)
```

### Streaming

```python
async for chunk in await client.aio.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

### Count Tokens and Compute Tokens

```python
response = client.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

#### Compute Tokens

Compute tokens is only supported in Vertex AI.

```python
response = client.models.compute_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

##### Async

```python
response = await client.aio.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

### Embed Content

```python
response = client.models.embed_content(
    model='text-embedding-004',
    contents='why is the sky blue?',
)
print(response)
```

```python
# multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['why is the sky blue?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality=10),
)

print(response)
```

### Imagen

#### Generate Images

Support for generate images in Gemini Developer API is behind an allowlist

```python
# Generate Image
response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()
```

#### Upscale Image

Upscale image is only supported in Vertex AI.

```python
# Upscale the generated image from above
response2 = client.models.upscale_image(
    model='imagen-3.0-generate-001',
    image=response1.generated_images[0].image,
    upscale_factor='x2',
    config=types.UpscaleImageConfig(
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response2.generated_images[0].image.show()
```

#### Edit Image

Edit image uses a separate model from generate and upscale.

Edit image is only supported in Vertex AI.

```python
# Edit the generated image from above
from google.genai.types import RawReferenceImage, MaskReferenceImage

raw_ref_image = RawReferenceImage(
    reference_id=1,
    reference_image=response1.generated_images[0].image,
)

# Model computes a mask of the background
mask_ref_image = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode='MASK_MODE_BACKGROUND',
        mask_dilation=0,
    ),
)

response3 = client.models.edit_image(
    model='imagen-3.0-capability-001',
    prompt='Sunlight and clear sky',
    reference_images=[raw_ref_image, mask_ref_image],
    config=types.EditImageConfig(
        edit_mode='EDIT_MODE_INPAINT_INSERTION',
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response3.generated_images[0].image.show()
```

### Veo

#### Generate Videos

Support for generate videos in Vertex and Gemini Developer API is behind an allowlist

```python
# Create operation
operation = client.models.generate_videos(
    model='veo-2.0-generate-001',
    prompt='A neon hologram of a cat driving at top speed',
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        fps=24,
        duration_seconds=5,
        enhance_prompt=True,
    ),
)

# Poll operation
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

video = operation.result.generated_videos[0].video
video.show()
```

## Chats

Create a chat session to start a multi-turn conversations with the model.

### Send Message

```python
chat = client.chats.create(model='gemini-2.0-flash-001')
response = chat.send_message('tell me a story')
print(response.text)
```

### Streaming

```python
chat = client.chats.create(model='gemini-2.0-flash-001')
for chunk in chat.send_message_stream('tell me a story'):
    print(chunk.text)
```

### Async

```python
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
response = await chat.send_message('tell me a story')
print(response.text)
```

### Async Streaming

```python
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
async for chunk in await chat.send_message_stream('tell me a story'):
    print(chunk.text)
```

## Files

Files are only supported in Gemini Developer API.

```cmd
!gsutil cp gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf .
!gsutil cp gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf .
```

### Upload

```python
file1 = client.files.upload(file='2312.11805v3.pdf')
file2 = client.files.upload(file='2403.05530.pdf')

print(file1)
print(file2)
```

### Get

```python
file1 = client.files.upload(file='2312.11805v3.pdf')
file_info = client.files.get(name=file1.name)
```

### Delete

```python
file3 = client.files.upload(file='2312.11805v3.pdf')

client.files.delete(name=file3.name)
```

## Caches

`client.caches` contains the control plane APIs for cached content

### Create

```python
if client.vertexai:
    file_uris = [
        'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
        'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
    ]
else:
    file_uris = [file1.uri, file2.uri]

cached_content = client.caches.create(
    model='gemini-1.5-pro-002',
    config=types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role='user',
                parts=[
                    types.Part.from_uri(
                        file_uri=file_uris[0], mime_type='application/pdf'
                    ),
                    types.Part.from_uri(
                        file_uri=file_uris[1],
                        mime_type='application/pdf',
                    ),
                ],
            )
        ],
        system_instruction='What is the sum of the two pdfs?',
        display_name='test cache',
        ttl='3600s',
    ),
)
```

### Get

```python
cached_content = client.caches.get(name=cached_content.name)
```

### Generate Content with Caches

```python
response = client.models.generate_content(
    model='gemini-1.5-pro-002',
    contents='Summarize the pdfs',
    config=types.GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)
print(response.text)
```

## Tunings

`client.tunings` contains tuning job APIs and supports supervised fine
tuning through `tune`.

### Tune

-   Vertex AI supports tuning from GCS source
-   Gemini Developer API supports tuning from inline examples

```python
if client.vertexai:
    model = 'gemini-1.5-pro-002'
    training_dataset = types.TuningDataset(
        gcs_uri='gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl',
    )
else:
    model = 'models/gemini-1.0-pro-001'
    training_dataset = types.TuningDataset(
        examples=[
            types.TuningExample(
                text_input=f'Input text {i}',
                output=f'Output text {i}',
            )
            for i in range(5)
        ],
    )
```

```python
tuning_job = client.tunings.tune(
    base_model=model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count=1, tuned_model_display_name='test_dataset_examples model'
    ),
)
print(tuning_job)
```

### Get Tuning Job

```python
tuning_job = client.tunings.get(name=tuning_job.name)
print(tuning_job)
```

```python
import time

running_states = set(
    [
        'JOB_STATE_PENDING',
        'JOB_STATE_RUNNING',
    ]
)

while tuning_job.state in running_states:
    print(tuning_job.state)
    tuning_job = client.tunings.get(name=tuning_job.name)
    time.sleep(10)
```

#### Use Tuned Model

```python
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents='why is the sky blue?',
)

print(response.text)
```

### Get Tuned Model

```python
tuned_model = client.models.get(model=tuning_job.tuned_model.model)
print(tuned_model)
```

### List Tuned Models

To retrieve base models, see [list base models](#list-base-models).

```python
for model in client.models.list(config={'page_size': 10, 'query_base': False}):
    print(model)
```

```python
pager = client.models.list(config={'page_size': 10, 'query_base': False})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

#### Async

```python
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}):
    print(job)
```

```python
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

### Update Tuned Model

```python
model = pager[0]

model = client.models.update(
    model=model.name,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    ),
)

print(model)
```


### List Tuning Jobs

```python
for job in client.tunings.list(config={'page_size': 10}):
    print(job)
```

```python
pager = client.tunings.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

#### Async

```python
async for job in await client.aio.tunings.list(config={'page_size': 10}):
    print(job)
```

```python
async_pager = await client.aio.tunings.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

## Batch Prediction

Only supported in Vertex AI.

### Create

```python
# Specify model and source file only, destination and job display name will be auto-populated
job = client.batches.create(
    model='gemini-1.5-flash-002',
    src='bq://my-project.my-dataset.my-table',
)

job
```

```python
# Get a job by name
job = client.batches.get(name=job.name)

job.state
```

```python
completed_states = set(
    [
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_PAUSED',
    ]
)

while job.state not in completed_states:
    print(job.state)
    job = client.batches.get(name=job.name)
    time.sleep(30)

job
```

### List

```python
for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
    print(job)
```

```python
pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

#### Async

```python
async for job in await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
):
    print(job)
```

```python
async_pager = await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
)
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

### Delete

```python
# Delete the job resource
delete_job = client.batches.delete(name=job.name)

delete_job
```

## Error Handling

To handle errors raised by the model service, the SDK provides this [APIError](https://github.com/googleapis/python-genai/blob/main/google/genai/errors.py) class.

```python
from google.genai import errors

try:
  client.models.generate_content(
      model="invalid-model-name",
      contents="What is your name?",
  )
except errors.APIError as e:
  print(e.code) # 404
  print(e.message)
```



================================================
FILE: CHANGELOG.md
================================================
# Changelog

## [1.11.0](https://github.com/googleapis/python-genai/compare/v1.10.0...v1.11.0) (2025-04-16)


### Features

* Add support for model_selection_config to GenerateContentConfig ([fdb0662](https://github.com/googleapis/python-genai/commit/fdb066288228ca101042ed9b11f783ed0d5f2799))
* Introduce json_schema quick accessor in Schema class to convert Google's Schema class into JSONSchema class. ([6e55222](https://github.com/googleapis/python-genai/commit/6e55222895a6639d41e54202e3d9a963609a391f))
* Support audio transcription in Vertex Live API ([9678aba](https://github.com/googleapis/python-genai/commit/9678ababb31282130e3cb9669e3670b627f91d86))
* Support configuring the underlying httpx client by allowing the caller to pass client arguments via HttpOptions. ([5130e0a](https://github.com/googleapis/python-genai/commit/5130e0a622210400d8d09399a06df55aa4af6a0e))
* Support RealtimeInputConfig, and language_code in SpeechConfig in python ([807f098](https://github.com/googleapis/python-genai/commit/807f098dedd0f885147fb10db7f79af9230999e0))
* Support user passing in async function to async generate_content and async generate_content_stream for automatic function calling ([33d190a](https://github.com/googleapis/python-genai/commit/33d190a77a198acfc857eff9677c88ed65e99758))
* Update VertexRagStore ([c4558e5](https://github.com/googleapis/python-genai/commit/c4558e5f65555d5a5352e4276c714392d594a3fa))


### Bug Fixes

* Get SSL_CERT_FILE or SSL_CERT_DIR environment variables for proper SSL handshake in API client. They are not automatically retrieved in httpx ([5782a5f](https://github.com/googleapis/python-genai/commit/5782a5f8a0bd1b3c741ea13b917d7d0091e9e12f))
* Update tests to use the pro 2.5 model gemini-2.5-pro-preview-03-25 ([fde4a8a](https://github.com/googleapis/python-genai/commit/fde4a8a484a5ab8cd0abadaaf108c552927f1976))

## [1.10.0](https://github.com/googleapis/python-genai/compare/v1.9.0...v1.10.0) (2025-04-09)


### ⚠ BREAKING CHANGES

* remove Part.from_video_metadata

### Features

* Add adapter size 2 for Gemini 2.0 Tuning ([959df89](https://github.com/googleapis/python-genai/commit/959df89e322efd4ed74f1c186a293b6f8fb7ee6e))
* Add domain to Web GroundingChunk ([9a75d48](https://github.com/googleapis/python-genai/commit/9a75d4885056771625f1af9ae735d61db9e7dc3c))
* Add session resumption. ([6e80ae7](https://github.com/googleapis/python-genai/commit/6e80ae77eba78607b91429168da20d618af9d3f0))
* Add thinking_budget to ThinkingConfig for Gemini Thinking Models ([71863e0](https://github.com/googleapis/python-genai/commit/71863e00c187c4eb9a379780f0a871235768a555))
* Add traffic type to GenerateContentResponseUsageMetadata ([925f983](https://github.com/googleapis/python-genai/commit/925f9836e1a6baf65adb5de5154870c1ec2621db))
* Add transcription support for MLDev ([c0a1b5c](https://github.com/googleapis/python-genai/commit/c0a1b5cdc169bfed61c69ffd8a08c0bffaaa80ce))
* Add types for configurable speech detection ([ae4ecee](https://github.com/googleapis/python-genai/commit/ae4ecee9562b71a292b61c63d33853568ab37f14))
* Add types to support continuous sessions with a sliding window ([7099e1e](https://github.com/googleapis/python-genai/commit/7099e1e99ce9e80a3b1080dcd1141a51e1990fea))
* Add UsageMetadata to LiveServerMessage ([018846a](https://github.com/googleapis/python-genai/commit/018846ae3e1f73d91522b8606e611152b7f63002))
* Added support for Context Window Compression ([e5c646c](https://github.com/googleapis/python-genai/commit/e5c646c106407a6c424a5d7fc6d022a395bac430))
* Populate X-Server-Timeout header when a request timeout is set. ([2af7b67](https://github.com/googleapis/python-genai/commit/2af7b67e811ae2b2e920c090006b2054193b404b))
* Remove experimental warnings for generate_videos and operations ([fa6007a](https://github.com/googleapis/python-genai/commit/fa6007ae9fb755e7cafb518985c96d54dd572a43))
* Remove experimental warnings from live api. ([007d1b1](https://github.com/googleapis/python-genai/commit/007d1b15e31000366352449c39679848dd7f622a))
* Support media resolution ([ef64f8a](https://github.com/googleapis/python-genai/commit/ef64f8a49171f2e05765ed7141d8ee51409a1ac7))


### Bug Fixes

* Remove Part.from_video_metadata ([c0947ab](https://github.com/googleapis/python-genai/commit/c0947ab20f75ed1c67985f7a2fdea04a5959de68))
* Upload file should support timeout (in milliseconds) configuration from http_options per request or from client ([5f3e895](https://github.com/googleapis/python-genai/commit/5f3e895276c94536ed797bcdca7fb913f95ddb01))


### Miscellaneous Chores

* Release 1.10.0 ([c136e41](https://github.com/googleapis/python-genai/commit/c136e4164d0c26530871c56653e21fc30caee511))

## [1.9.0](https://github.com/googleapis/python-genai/compare/v1.8.0...v1.9.0) (2025-04-01)


### Features

* Add specialized `send` methods to the live api ([9c4e4dc](https://github.com/googleapis/python-genai/commit/9c4e4dcdff508230f635c5d174486b08b87f86c9))
* Expose generation_complete, input/output_transcription & input/output_audio_transcription to SDK for Vertex Live API ([e5685ad](https://github.com/googleapis/python-genai/commit/e5685adc7a064b511ddb490ef0aaf27f3070775f))
* Merge GenerationConfig into LiveConnectConfig ([d22535e](https://github.com/googleapis/python-genai/commit/d22535e700178f27e458a02868cd2c06d5470e34))


### Bug Fixes

* Make response arg in APIError class constructor optional. [#572](https://github.com/googleapis/python-genai/issues/572) ([7b3f4a4](https://github.com/googleapis/python-genai/commit/7b3f4a4c50646c29293c0196b471b7bf3a29f102))


### Documentation

* Docstring improvements ([77f5356](https://github.com/googleapis/python-genai/commit/77f53566bbff3a715d2c7e5e83ada61ffd80ac96))

## [1.8.0](https://github.com/googleapis/python-genai/compare/v1.7.0...v1.8.0) (2025-03-26)


### Features

* Add engine to VertexAISearch ([21f0394](https://github.com/googleapis/python-genai/commit/21f03941e23173d5c4bfce8551e9f64410a4cd95))
* Add IMAGE_SAFTY enum value to FinishReason ([3a65fb0](https://github.com/googleapis/python-genai/commit/3a65fb0e841b183d42023aef37e22207cdc9829f))
* Add MediaModalities for ModalityTokenCount ([fb2509c](https://github.com/googleapis/python-genai/commit/fb2509c471a42a3144d7b25e63be54d32608851c))
* Add Veo 2 generate_videos support in Go SDK ([55b2923](https://github.com/googleapis/python-genai/commit/55b2923de0b925ad5487d1584331cb093773a325))
* Allow title property to be sent to Gemini API. Gemini API now supports the title property, so it's ok to pass this onto both Vertex and Gemini API. ([f2f92a7](https://github.com/googleapis/python-genai/commit/f2f92a739a764f1eae20e932fdea6a3305305f77))
* **chats:** Allow user to create chat session with list of ContentDict ([43c5379](https://github.com/googleapis/python-genai/commit/43c5379aa0a6da1f742ab0eada72cbff0b2bed40)), closes [#467](https://github.com/googleapis/python-genai/issues/467)
* Move set event loop into try except logic when setting auth lock ([d04b6a6](https://github.com/googleapis/python-genai/commit/d04b6a65129960d6e70acec2879afbd8f7c5b0e0))
* Save prompt safety attributes in dedicated field for generate_images ([e5bbb0e](https://github.com/googleapis/python-genai/commit/e5bbb0e6b77f12e7e6877ae0e976fda3f8beb604))
* Support new UsageMetadata fields ([122cdc8](https://github.com/googleapis/python-genai/commit/122cdc86f1c7381f04f6c8ab3ea86409fcb85661))


### Bug Fixes

* Improve logging for response.parsed (fixes [#455](https://github.com/googleapis/python-genai/issues/455)) ([64012dd](https://github.com/googleapis/python-genai/commit/64012dd6a6c1877114100f52c10a45a7f7ff885e))
* Schema transformer logic fix. ([f64bcba](https://github.com/googleapis/python-genai/commit/f64bcba552156c1f344f8c03932a24c0a4f9c222))
* Set event loop before asyncio.Lock() to ensure threading safety ([be2d9c6](https://github.com/googleapis/python-genai/commit/be2d9c61176f8330166e7912e233fb07fa82e4a8))
* Surface complete error details from backend ([38f5beb](https://github.com/googleapis/python-genai/commit/38f5bebba79182f78c4aa921861151626c79a84d))
* **chats:** Raise error when `types.Content` is passed to `send_message()` to correctly adhere to type annotation. To migrate code previously passing `content = types.Content(...)` to `send_message()`, pass `content.parts` instead.

### Documentation

* Log warning to users that Part.from_video_metadata will be deprecated. ([2d12f54](https://github.com/googleapis/python-genai/commit/2d12f544bc3f8d1e2720855f0fe3519881baaeb0))

## [1.7.0](https://github.com/googleapis/python-genai/compare/v1.6.0...v1.7.0) (2025-03-18)


### Features

* Bump up the websockets version for proxy support ([b996c4b](https://github.com/googleapis/python-genai/commit/b996c4b62be8c4d108dbe572372a2c31661e9bcc))
* Leverage httpx connection pooling and avoid instantiation of httpx.Client or httpx.AsyncClient for each call ([2ac5129](https://github.com/googleapis/python-genai/commit/2ac5129d4005d797f29df0a213e2455d7d730b43))


### Bug Fixes

* **chats:** Fix duplicate history when appending AFC history ([e4b1d8a](https://github.com/googleapis/python-genai/commit/e4b1d8af03be5579ff6cb0ba42dd0d3d65126892))
* **chats:** Raise error when `types.Content` is passed to `send_message()` to correctly adhere to type annotation. To migrate code previously passing `content = types.Content(...)` to `send_message()`, pass `content.parts` instead.
* Fix incorrect casing of upload status and url headers in Files API. httpx.Response returns 'x-goog-upload-status', but we were checking for 'X-Goog-Upload-Status'. Also clean up upload_file and download_file requests. ([2ac5129](https://github.com/googleapis/python-genai/commit/2ac5129d4005d797f29df0a213e2455d7d730b43))


### Documentation

* Use consistent terminology for Gemini Developer API and Vertex AI ([7f1cc22](https://github.com/googleapis/python-genai/commit/7f1cc22e6220d4550726bd75c0dff922698d674e))

## [1.5.0](https://github.com/googleapis/python-genai/compare/v1.4.0...v1.5.0) (2025-03-07)


### Features

* Determine mime_type from local images ([5e84ddc](https://github.com/googleapis/python-genai/commit/5e84ddc8d4265069e6a03c9a99ff6891cc641297))
* Enable generate_videos for Gemini Developer API ([4a242f6](https://github.com/googleapis/python-genai/commit/4a242f6f759a5e3fd1435b548d8dda38091b9cee))
* Enable image to video for generate_videos ([787354b](https://github.com/googleapis/python-genai/commit/787354bf96d7a4fc479d9e8bd432e76ab54ab54c))
* Expand files.download to work on Video and GeneratedVideo objects ([8d4d6fd](https://github.com/googleapis/python-genai/commit/8d4d6fd6271b3f74d3efae676fc08418b9bb5cda))
* Support asynchronously upload and download files using httpx ([498c01d](https://github.com/googleapis/python-genai/commit/498c01da1d2820b97b2218022b7e42840453e739))


### Bug Fixes

* Fix incorrect unit for `timeout_in_seconds` in HttpOptions. ([a9be9a2](https://github.com/googleapis/python-genai/commit/a9be9a20691249bacf162d97ef6664883102edc7))
* Fix Video.show() when uri and video_bytes are provided ([3477c40](https://github.com/googleapis/python-genai/commit/3477c40cf89eb8006b6a0877710ec77cdb1edeff))

## [1.4.0](https://github.com/googleapis/python-genai/compare/v1.3.0...v1.4.0) (2025-03-05)


### Features

* Add response_id and create_time to GenerateContentResponse ([b46ed36](https://github.com/googleapis/python-genai/commit/b46ed361d9c0845880a1447ee42dd3c0f6f7a886))
* Allow non-content types in generate_content ([cbaaf4a](https://github.com/googleapis/python-genai/commit/cbaaf4ae19e17f4fafb48e8671c10786095e2936))
* Enable Live API initial connect to accept functions directly, in addition to just FunctionDeclaration ([91b1d3e](https://github.com/googleapis/python-genai/commit/91b1d3ee85fd32e9243e3a2da4d10a763e4d0005))
* Enable minItem, maxItem, nullable for Schema type when calling Gemini API. ([867ce70](https://github.com/googleapis/python-genai/commit/867ce70cf4a7ebd52554af3d494f73fd3cd4f6b6))
* Implement get history to return comprehensive or curated chat history ([92deda1](https://github.com/googleapis/python-genai/commit/92deda1b86c27c4f93691d4f4056ed64ba84d6a8))
* Support aspect ratio for edit_image ([5423a58](https://github.com/googleapis/python-genai/commit/5423a58f3abea2b468973c0c071ced6547f6cef1))


### Bug Fixes

* Allow user do batch generate content when passing a list of strings ([cbaaf4a](https://github.com/googleapis/python-genai/commit/cbaaf4ae19e17f4fafb48e8671c10786095e2936))
* Fix chats.send_message_stream curated history ([bcf2be0](https://github.com/googleapis/python-genai/commit/bcf2be03ae6d51957052cacfb8ed905045cc6f77))
* Log warn instead of raise error for GenerateContentResponse.text quick accessor when there are mixed part types ([55a0638](https://github.com/googleapis/python-genai/commit/55a0638075d89b873aea581e2012a0951cbcf7fb))
* Remove the keyword parameter requirement for UserContent and ModelContent ([0cc292f](https://github.com/googleapis/python-genai/commit/0cc292f5b61dcc2756c2110038c93c0c23b29c1d))

## [1.3.0](https://github.com/googleapis/python-genai/compare/v1.2.0...v1.3.0) (2025-02-24)


### Features

* Add generate_videos (Veo 2) support for Python ([e9e2be7](https://github.com/googleapis/python-genai/commit/e9e2be73d5a2f6a03d6a2a665cf35e96f2ed97cd))
* Add sdk logger instance (fixes [#278](https://github.com/googleapis/python-genai/issues/278)) ([cf281b5](https://github.com/googleapis/python-genai/commit/cf281b58195be1dd374e4754e432603ac215202f))
* Introduce response.executable_code and response.code_execution_result quick accessors for GenerateContentResponse class ([3725ddf](https://github.com/googleapis/python-genai/commit/3725ddf44a0df04f6fac0bd5295f45ed20b4a8fe))
* Introduce UserContent and ModelContent to facilitate easier content creation ([c8cfef8](https://github.com/googleapis/python-genai/commit/c8cfef85ca1f7901e802800f580915a14340cae7))
* Native async client support using httpx ([c38da8d](https://github.com/googleapis/python-genai/commit/c38da8d4425a92bd6ba483abccc3dbeafbf8daa4))
* Provide a public property for determining the module backend. ([8e561de](https://github.com/googleapis/python-genai/commit/8e561de04965bb8766db87ad8eea7c57c1040442))


### Bug Fixes

* Enable sending empty input to Live API as turn complete ([99a5510](https://github.com/googleapis/python-genai/commit/99a55100e1c4e91f53b22881fbff67e62f6fb99d))
* Fix automatic function calling warning message logic. ([b99da95](https://github.com/googleapis/python-genai/commit/b99da95ae0b986d77853c917931bdfa94c6b7714))
* Fix duplicate get_function_response_parts in (async) generate_content_stream and ensure chunk isn't empty before get_function_response_parts ([d4193e6](https://github.com/googleapis/python-genai/commit/d4193e6eb22d62c61b2727600171eb06780bfa18))
* Properly handle empty json response with headers for list models ([859ebc3](https://github.com/googleapis/python-genai/commit/859ebc3dc51aab9834b9034b5a1a205bcc72d25d))


### Documentation

* Add docs for error handling ([#317](https://github.com/googleapis/python-genai/issues/317)) ([6e1cb82](https://github.com/googleapis/python-genai/commit/6e1cb82704f6cb05b67191a319190c5ce0f67d9c))
* Add instruction on how to disable automatic function calling. ([f8b12d5](https://github.com/googleapis/python-genai/commit/f8b12d53567585094f7e7b1d015b89c993f480de))
* Regenerate docs for 1.2.0 ([30a3493](https://github.com/googleapis/python-genai/commit/30a34931d200feb12da54e978ea1f4a4a3d1e82e))
* Remove negative_prompt from image samples (fixes [#339](https://github.com/googleapis/python-genai/issues/339)) ([81e18f0](https://github.com/googleapis/python-genai/commit/81e18f053048ff4b3d2c5dde0726234d51cc8290))
* Update docs for models modules ([d96bba2](https://github.com/googleapis/python-genai/commit/d96bba29326113dba46ff16932deeb50618b8d8c))

## [1.2.0](https://github.com/googleapis/python-genai/compare/v1.1.0...v1.2.0) (2025-02-12)


### Features

* Enable Media resolution for Gemini API. ([6cdf61d](https://github.com/googleapis/python-genai/commit/6cdf61d09f0dec0b27f2be6a0487ec5f4792d62f))
* Support property_ordering in response_schema (fixes [#236](https://github.com/googleapis/python-genai/issues/236)) ([01b15e3](https://github.com/googleapis/python-genai/commit/01b15e32d3823a58d25534bb6eea93f30bf82219))


### Bug Fixes

* Default to list base models for async models list ([d3226b7](https://github.com/googleapis/python-genai/commit/d3226b7ce9fde115e503da7f9108d735fc89325c))
* Remove Type import from types.py that get's shadowed by API Type type. (fixes [#310](https://github.com/googleapis/python-genai/issues/310)) ([78f58c3](https://github.com/googleapis/python-genai/commit/78f58c3f63ec1d9fb916e81f9b962d5b65b13ec2))
* Use typing_extensions.TypedDict for TypedDict types (fixes [#189](https://github.com/googleapis/python-genai/issues/189)) ([996562a](https://github.com/googleapis/python-genai/commit/996562afed6d19b20bef343262e4c8820559c2d6))


### Documentation

* Client initialization using environmental variables. ([e4c2ffc](https://github.com/googleapis/python-genai/commit/e4c2ffcdb5ddd68ded5f3e2b98c0b44afe91ca1b))
* Fix File.expiration_time description (fixes [#318](https://github.com/googleapis/python-genai/issues/318)) ([729f619](https://github.com/googleapis/python-genai/commit/729f619bdc88f0d775fda3b9f1a75a527ddf90ac))
* Fix files.upload docs to use 'file' instead of 'path' (fixes [#306](https://github.com/googleapis/python-genai/issues/306)) ([2b35d0c](https://github.com/googleapis/python-genai/commit/2b35d0ca4e74b67f98e64da88286148370be9b6f))

## [1.1.0](https://github.com/googleapis/python-genai/compare/v1.0.0...v1.1.0) (2025-02-10)


### Features

* Add support for typing.Literal in response_schema (fixes [#264](https://github.com/googleapis/python-genai/issues/264)) ([384c4eb](https://github.com/googleapis/python-genai/commit/384c4eb7bac7ac867db8f19777c33e01daff89ef))
* Allow converting a function into FunctionDeclaration with api option ([408e28f](https://github.com/googleapis/python-genai/commit/408e28fc9892996eb1cb617de3cf7ce658deed16))
* Support generate content config for each chat turn ([ca19100](https://github.com/googleapis/python-genai/commit/ca1910026d7e425c1f9adb13779ec43bc2b9d99e))
* Tuning - `Tuning.tune` for Gemini API no longer blocks until the tuning operation is resolved ([fcf8888](https://github.com/googleapis/python-genai/commit/fcf88881698eabfa7d808df0f1353aa6bcc54cb8))


### Bug Fixes

* Remove duplicate function invocations and ensure automatic function calling can be disabled in generate_content_stream ([0958fbe](https://github.com/googleapis/python-genai/commit/0958fbe4ce68c09a6b665fcacb2e3a16dfd822d2))
* Response_schema generation for pydantic fields with type Optional[list] and Optional[MyBaseModel] now work correctly (fixes [#246](https://github.com/googleapis/python-genai/issues/246)) ([8330561](https://github.com/googleapis/python-genai/commit/833056195bb7fe7c7e45095df2eea748f48de49c))
* Support dict response_schema with 'any_of' key ([75f5056](https://github.com/googleapis/python-genai/commit/75f505637c89df50120e3ea25c6379fa49b54abf))
* Common - Do not fail when server returns unknown enum values on Python &lt;3.11 ([843d86d](https://github.com/googleapis/python-genai/commit/843d86d38699be5f6acc58e5e5a915acab8018be))


### Documentation

* Add docs on how to structure `contents` (fixes [#274](https://github.com/googleapis/python-genai/issues/274)) ([da356cb](https://github.com/googleapis/python-genai/commit/da356cb12d9b68fe9ccae0fec1c059399e7f9685))
* Regenerate docs for 1.0 ([fbee816](https://github.com/googleapis/python-genai/commit/fbee81625178fe9d39c7856bcae01ebf570b154c))
* Update model names in README, fix function calling example ([11c0274](https://github.com/googleapis/python-genai/commit/11c0274fbac5c82e39dac4e99ac7237cd9705e0b))

## [1.0.0](https://github.com/googleapis/python-genai/compare/v0.8.0...v1.0.0) (2025-02-05)


### ⚠ BREAKING CHANGES

* Remove deprecated field: `deprecated_response_payload`

### Features

* Add labels for GenerateContent requests ([3e3b82d](https://github.com/googleapis/python-genai/commit/3e3b82dcdc8d50852a9fdd452c8e0957bb4493de))
* Add Union support to response_schema ([eedf4f1](https://github.com/googleapis/python-genai/commit/eedf4f12aea9f5a0d4056d6f14a91b14db1903cf))
* Support automatic function calling in models.generate_content_stream and send_message_stream, sync and async mode ([7c9f8b5](https://github.com/googleapis/python-genai/commit/7c9f8b5833e393e5879fd267abf3a34d122a5d1d))


### Bug Fixes

* Avoid false test failure by skipping incompatible test case for python 3.13 ([7f10e55](https://github.com/googleapis/python-genai/commit/7f10e550db4767ab172f5f5b00eda484cb0141d6))
* Default to list base models (instead of tuned models) ([ef48f0d](https://github.com/googleapis/python-genai/commit/ef48f0dfcae1811ddc3160e368a5ef25a535ee25))
* Handle pydantic model type recognition gracefully for python 3.9 and 3.10 ([4a38fdf](https://github.com/googleapis/python-genai/commit/4a38fdf53fd689fdab611dbcaebc9570c7d46bd5))
* Raise error when Gemini API response_schema has 'default' or 'anyof' fields ([c50bca0](https://github.com/googleapis/python-genai/commit/c50bca00c814b39a98fc6aacba0c4c429fed0570))
* Remove redundant contents in Automatic Function Calling history ([5510595](https://github.com/googleapis/python-genai/commit/5510595eb18ad0a0af34ce63df1685b49e03ce10))
* Remove unsupported parameter from Gemini API ([f8addb5](https://github.com/googleapis/python-genai/commit/f8addb544e545100ae7342621910a2338072ddc6))


### Documentation

* Remove experimental classification from description. ([b14ac57](https://github.com/googleapis/python-genai/commit/b14ac57b07d4e7794d83f2b2123a4df80890f772))
* Remove thoughts examples and update tests ([af3b339](https://github.com/googleapis/python-genai/commit/af3b339a9d58e25cb3baa6fd3d0b1d078c620f9d))
* Update documentation on how to set api version using genai client ([6fd4425](https://github.com/googleapis/python-genai/commit/6fd442520d1b59003fd5fcd24d395b9790caec6f))
* Update instruction on function calling experience in `ANY` mode. ([451cf98](https://github.com/googleapis/python-genai/commit/451cf989e2a917884475cf66fce1d809a7495b53))


### Code Refactoring

* Remove deprecated field: `deprecated_response_payload` ([197fa46](https://github.com/googleapis/python-genai/commit/197fa46c9639799b8d71ddf92ab372140dc5b65b))

## [0.8.0](https://github.com/googleapis/python-genai/compare/v0.7.0...v0.8.0) (2025-01-30)


### ⚠ BREAKING CHANGES

* Rename files.upload argument to "file" instead of "path".

### Features

* Add enhanced_prompt to GeneratedImage class ([76c810b](https://github.com/googleapis/python-genai/commit/76c810b6bb82bde4cc4fe9754e0bbcc85e10a4c4))
* Added Operation and PredictOperation (internal module) ([309dd26](https://github.com/googleapis/python-genai/commit/309dd26cb3aa761bfae7070ffa5eab23b2a5a75c))
* Support global endpoint natively in Vertex ([f4530b0](https://github.com/googleapis/python-genai/commit/f4530b0af972d0b5e376e7463af235b8378e0694))
* Support unknown enum values ([da448b3](https://github.com/googleapis/python-genai/commit/da448b3cb2bf5d6cba6fdefa8d15365253fe0384))


### Bug Fixes

* Streaming in Vertex AI Express ([ff78b7b](https://github.com/googleapis/python-genai/commit/ff78b7b0277800e2b4b9e9958a87688383422d29))


### Documentation

* Correct generate content with uploaded file example ([8cea052](https://github.com/googleapis/python-genai/commit/8cea052ac148890a343f14b7185e394646802b41))


### Miscellaneous Chores

* Rename files.upload argument to "file" instead of "path". ([f68aa1f](https://github.com/googleapis/python-genai/commit/f68aa1f63f04629f9eb8629c5b1359f500f89a47))

## [0.7.0](https://github.com/googleapis/python-genai/compare/v0.6.0...v0.7.0) (2025-01-28)


### ⚠ BREAKING CHANGES

* Remove skip_project_and_location_in_path from async models.list
* remove "tuning.distill"
* Change asynchronous streaming output to an awaitable async iterator for client.aio.models.generate_content_stream and client.aio.chat.send_message_stream
* Remove pillow as a required dependency
* rename batches.list method signature.
* make Part, FunctionDeclaration, Image, and GenerateContentResponse classmethods argument keyword only
* Remove skip_project_and_location_in_path from HttpOption
* Renamed FunctionDeclaration class functions to reflect the fact that they work off of the `Callable` type not just functions.
* Moved the HttpOptions class into types.py, by making it autogenerated. This causes the following breaking change:
* rename generate_image() to generate_images(), rename GenerateImageConfig to GenerateImagesConfig, rename GenerateImageResponse to GenerateImagesResponse, rename GenerateImageParameters to GenerateImagesParameters

### Features

* [genai-modules][models] Add HttpOptions to all method configs for models. ([76fdde7](https://github.com/googleapis/python-genai/commit/76fdde7138b7bc3bd1e761bca753610fbb83a1af))
* Add support for enhance_prompt to model.generate_image ([d09e14e](https://github.com/googleapis/python-genai/commit/d09e14ea77de455c545c07ef0e4d0735c21521af))
* Added support for the new HttpOptions class in the per request options overrides. ([ad57025](https://github.com/googleapis/python-genai/commit/ad5702512ee78ad7bcda6233a7d1eafaaaf50e1f))
* Change asynchronous streaming output to an awaitable async iterator for client.aio.models.generate_content_stream and client.aio.chat.send_message_stream ([0c124eb](https://github.com/googleapis/python-genai/commit/0c124eba8909e77336a7c22681d57080d6f1a6ad))
* Enable enum support in the GenerateContentConfig.response_schema ([fe82e10](https://github.com/googleapis/python-genai/commit/fe82e1065095eabfe97cd197ebcea5b73efb01cd))
* Handle a wider variety of response_schemas - primitives and nested lists. ([24fffea](https://github.com/googleapis/python-genai/commit/24fffea93a6c127826a16a4183ff38a4376a4d5f))
* Images - Added Image.mime_type ([71a8f1d](https://github.com/googleapis/python-genai/commit/71a8f1da1904a34c9f8c720d2356e50db7b279b3))
* Make Part, FunctionDeclaration, Image, and GenerateContentResponse classmethods argument keyword only ([b58f4e0](https://github.com/googleapis/python-genai/commit/b58f4e03050fce29a6c47091cf25ef6f440c2e7e))
* Remove skip_project_and_location_in_path from async models.list ([d788626](https://github.com/googleapis/python-genai/commit/d788626a8de8bd8540739b673ae34530b494d83e))
* Remove skip_project_and_location_in_path from HttpOption ([a1c0435](https://github.com/googleapis/python-genai/commit/a1c043521558aca80a3bf55ff3eb96c95ec12e72))
* Support GenerateContentResponse.parsed to return Enum ([e214211](https://github.com/googleapis/python-genai/commit/e2142118ae7796c52503f327ca8d949a80be0e16))
* Validate enum value for different backend endpoint. ([97bb958](https://github.com/googleapis/python-genai/commit/97bb9580511c3517134d867500c8d155c231e04c))


### Bug Fixes

* Add required key to nested fields in function declaration schemas ([c7dba47](https://github.com/googleapis/python-genai/commit/c7dba473610547f153f13f82b8d8977a60b11308))
* Fix pydantic list support in Python 3.9 and 3.10 (fixes [#52](https://github.com/googleapis/python-genai/issues/52)) ([81150b3](https://github.com/googleapis/python-genai/commit/81150b3e0b21eded0cbdb2d0e3308b7866de619a))
* Handle nullable types in pydantic classes (fixes [#62](https://github.com/googleapis/python-genai/issues/62)) ([773e1c0](https://github.com/googleapis/python-genai/commit/773e1c0e28ff6dbfc799275c24b280c79375f9d5))
* Support parameterized generics Union type in automatic function calling parameters (fixes [#22](https://github.com/googleapis/python-genai/issues/22)) ([04475ab](https://github.com/googleapis/python-genai/commit/04475ab9d75e38f33ed07e7c6ad03bde36634105))


### Documentation

* Add examples & tests for thinking model ([59e2763](https://github.com/googleapis/python-genai/commit/59e27633de46f3521916a6d7819eae6ffc387405))
* Correct usage ([0c546de](https://github.com/googleapis/python-genai/commit/0c546deee62b5bbfbf3e05938f7b46a89861250e))
* Document experimental state of the live module. ([115ac47](https://github.com/googleapis/python-genai/commit/115ac4769088c43ef9eafe5eff588abc340b8213))
* Fix Blob type docstring. ([7f38e55](https://github.com/googleapis/python-genai/commit/7f38e55207d0c049a530e1f5eb605a251c133382))
* Remove repeated docstring for interrupted in docs ([230cfbc](https://github.com/googleapis/python-genai/commit/230cfbc73e45c0ea9a1a360890a68831a0dc3b60))
* Remove the word "class" in docstring and sync Live types docstring up-to-date ([7fa3ef3](https://github.com/googleapis/python-genai/commit/7fa3ef335e08cc752fa1eb41bf7d130ffccf7898))
* Update supported model parameter format for generate_content ([a5c3a1c](https://github.com/googleapis/python-genai/commit/a5c3a1ce6b0cffe3a91ffcb3756b3176ecf7b213))


### Code Refactoring

* Moved the HttpOptions class into types.py, by making it autogenerated. This causes the following breaking change: ([ad57025](https://github.com/googleapis/python-genai/commit/ad5702512ee78ad7bcda6233a7d1eafaaaf50e1f))
* Remove "tuning.distill" ([ca8cd5e](https://github.com/googleapis/python-genai/commit/ca8cd5e389263a261d1d9e05e47d6f108803efff))
* Remove pillow as a required dependency ([8ccdf2e](https://github.com/googleapis/python-genai/commit/8ccdf2e99246ba948b97f786596ae418f85749b0))
* Rename batches.list method signature. ([aa7c071](https://github.com/googleapis/python-genai/commit/aa7c071cb5922510f189da7875e1a3f6e1836733))
* Rename generate_image() to generate_images(), rename GenerateImageConfig to GenerateImagesConfig, rename GenerateImageResponse to GenerateImagesResponse, rename GenerateImageParameters to GenerateImagesParameters ([65d7bf5](https://github.com/googleapis/python-genai/commit/65d7bf5f519570c7af72d434a22e302fbafe0735))
* Renamed FunctionDeclaration class functions to reflect the fact that they work off of the `Callable` type not just functions. ([0e4a003](https://github.com/googleapis/python-genai/commit/0e4a0030295323177b38d052f7b7d695c07555b2))

## [0.6.0](https://github.com/googleapis/python-genai/compare/v0.5.0...v0.6.0) (2025-01-21)


### Features

* Add support for audio_timestamp to types.GenerateContentConfig (fixes [#132](https://github.com/googleapis/python-genai/issues/132)) ([116395b](https://github.com/googleapis/python-genai/commit/116395b23d2415407c01efb30cb6c1863a4a3032))
* Add support for case insensitive enum types. (fixes [#11](https://github.com/googleapis/python-genai/issues/11)) ([252f302](https://github.com/googleapis/python-genai/commit/252f302d3bb02271ba3e8953aeb4fb8fa7023fa6))
* Add support for class methods in the Tools for the GenerateContent Function ([5afa6bb](https://github.com/googleapis/python-genai/commit/5afa6bb9ac9445491bea6af28655535c78976a49))
* Add ThinkingConfig to generate content config. ([c23c42c](https://github.com/googleapis/python-genai/commit/c23c42c1bf012b36717dd21e86fdb7859b43b759))
* Implement client.files.download ([a034b77](https://github.com/googleapis/python-genai/commit/a034b77dc6c0806d791942d2a6ced4cf5973d2c3))
* Support BytesIO when uploading files ([4b490dc](https://github.com/googleapis/python-genai/commit/4b490dce9ead29fe28411b20801fc95a8617143c))
* Support lists in response_schema ([8ed933d](https://github.com/googleapis/python-genai/commit/8ed933dd4068add6983fc084df1cb434b444ba2a))
* Usability change to simplify generate content with uploaded files ([5c372ae](https://github.com/googleapis/python-genai/commit/5c372ae61914c75a5700297a8f2301f76379e926))


### Bug Fixes

* Fix count_tokens system_instruction and tools config ([056eaba](https://github.com/googleapis/python-genai/commit/056eabad0cd863d5729f423416a4961c5113c1a5))
* Fix models.list() with empty tuned models ([75409f1](https://github.com/googleapis/python-genai/commit/75409f12d9f531f126c427b3bd9a0b8d3bacabf7))
* Fixed the `bytes` type handling (the `base64.urlsafe_bencode` error) ([1bc161d](https://github.com/googleapis/python-genai/commit/1bc161d81c78afa35f21a24ee1840b5ed03f3a96))
* Project and location are required when using client in Vertex AI mode. ([d5859fa](https://github.com/googleapis/python-genai/commit/d5859fa8d89bb26b3c31ee3dab0deaf9c159e615))

### Documentation

* Add project description to README. ([384f413](https://github.com/googleapis/python-genai/commit/384f413666e8d2430a943163dd109814ee9b6137))
* Update formatting in README.md ([d33fb37](https://github.com/googleapis/python-genai/commit/d33fb37a082e04f512063d88bbe500a9acdbfb2c))
* Update models.list doc and code examples to include list base models ([f2e0c43](https://github.com/googleapis/python-genai/commit/f2e0c43417f4b6a7ca8b1b0e5bef97f68a8ff84b))
* Tool use example in docs ([87fe5b0](https://github.com/googleapis/python-genai/commit/87fe5b0cdfbf34c295398217185d857ca1413c02))
* Tool use example in README.md ([87fe5b0](https://github.com/googleapis/python-genai/commit/87fe5b0cdfbf34c295398217185d857ca1413c02))


## [0.5.0](https://github.com/googleapis/python-genai/compare/v0.4.0...v0.5.0) (2025-01-13)


### Features

* Support API keys for VertexAI mode generate_content ([0e4b0e5](https://github.com/googleapis/python-genai/commit/0e4b0e5ee0ecfef34f6576d8aab24cf0554bbd85))
* Support list models to return base models ([0f713f1](https://github.com/googleapis/python-genai/commit/0f713f177e84ab7a2071c635ec7231ee4fbf4657))
* Support parsing 'interrupted' field in Live Python SDK. ([eab2c4a](https://github.com/googleapis/python-genai/commit/eab2c4a9513d4ce58270b82e10698a945c01aaca))
* Use `ser_json_byte` `val_json_bytes` in bytes type public interface ([1176e43](https://github.com/googleapis/python-genai/commit/1176e43c2f87edf5566512b8f3c433b77684f87b))


### Bug Fixes

* Update header type ([62c45f9](https://github.com/googleapis/python-genai/commit/62c45f9ecdb0e7bc539d22a967672762fa6c50a8))


### Documentation

* Correct description of path parameter that only path-like object is supported ([336498b](https://github.com/googleapis/python-genai/commit/336498baebee2e064e2f44d9c6d96903bcfd63bf))

## [0.4.0](https://github.com/googleapis/python-genai/compare/v0.3.0...v0.4.0) (2025-01-08)


### ⚠ BREAKING CHANGES

* Make Imagen upscale_factor a required argument, make upscale config optional

### Features

* ApiClient - support timeout in HttpOptions ([b22286f](https://github.com/googleapis/python-genai/commit/b22286fa3cd48a7918ffa2acaa40e53f3c8bb5d8))
* Enable response_logprobs and logprobs for Google AI ([5586f3d](https://github.com/googleapis/python-genai/commit/5586f3d650964547c3e7cafa1165f9c7d1306529))
* Support function_calls quick accessor in GenerateContentResponse class ([81b8a23](https://github.com/googleapis/python-genai/commit/81b8a236b85081aa02fc4e38e27189bc2deb5cfb))


### Bug Fixes

* Change string type to int type ([5c15243](https://github.com/googleapis/python-genai/commit/5c1524370214128752905faefcf2690592b2dc90))
* FunctionCallCancellation ids type. ([b0e46b7](https://github.com/googleapis/python-genai/commit/b0e46b72aeef938414a68efd773bdac0b25c05d2))
* Gracefully catch error if streamed json does not meet schema validation (fixes [#14](https://github.com/googleapis/python-genai/issues/14)) ([f494432](https://github.com/googleapis/python-genai/commit/f494432900d60e64cbef69424918904ddbe255b1))
* Update RealtimeClientLiveMessage realtime content parameter field. ([4340939](https://github.com/googleapis/python-genai/commit/4340939d17ee38cad0d8d64aafcde5c3ef89f78f))


### Documentation

* Add readme example for Chats module ([830f0f5](https://github.com/googleapis/python-genai/commit/830f0f56e6663c173454a47ca97867031bd89819))
* Fix typo in code document ([f9243e8](https://github.com/googleapis/python-genai/commit/f9243e890aa42ae13b6f83dd885824b629a17cd3))
* Fix typo in CONTRIBUTING.md ([3e42644](https://github.com/googleapis/python-genai/commit/3e42644784304d45d0b0bfdc8279958109650576))


### Code Refactoring

* Make Imagen upscale_factor a required argument, make upscale config optional ([b06629f](https://github.com/googleapis/python-genai/commit/b06629f879b548a209e8517e92f5923d1f25c3a8))

## [0.3.0](https://github.com/googleapis/python-genai/compare/v0.2.2...v0.3.0) (2024-12-17)


### BREAKING CHANGES
* contents must be passed to CreateCachedContentConfig instead as a parameter to create_cached_content.

### Features

* Add support for Pydantic default value in Automatic Function Calling.
* Add support for thought.
* Add support for streaming chat.



================================================
FILE: CONTRIBUTING.md
================================================
# Contributing

The Google Gen AI SDK will accept contributions in the future.



================================================
FILE: LICENSE
================================================

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.



================================================
FILE: MANIFEST.in
================================================
include LICENSE
include README.md


================================================
FILE: pyproject.toml
================================================
[build-system]
requires = ["setuptools", "wheel", "twine>=6.1.0", "packaging>=24.2", "pkginfo>=1.12.0"]

[project]
name = "google-genai"
version = "1.11.0"
description = "GenAI Python SDK"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.9"
authors = [
    { name = "Google LLC", email = "googleapis-packages@google.com" },
]
classifiers = [
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Internet",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    "anyio>=4.8.0, <5.0.0",
    "google-auth>=2.14.1, <3.0.0",
    "httpx>=0.28.1, <1.0.0",
    "pydantic>=2.0.0, <3.0.0",
    "requests>=2.28.1, <3.0.0",
    "websockets>=13.0.0, <15.1.0",
    "typing-extensions>=4.11.0, <5.0.0",
]

[project.urls]
Homepage = "https://github.com/googleapis/python-genai"

[tool.setuptools]
packages = [
    "google",
    "google.genai",
]
include-package-data = true


================================================
FILE: requirements.txt
================================================
absl-py==2.1.0
annotated-types==0.7.0
anyio==4.8.0
cachetools==5.5.0
certifi==2024.8.30
charset-normalizer==3.4.0
coverage==7.6.9
httpx==0.28.1
google-auth==2.37.0
idna==3.10
iniconfig==2.0.0
packaging==24.2
pillow==11.0.0
pluggy==1.5.0
pyasn1==0.6.1
pyasn1_modules==0.4.1
pydantic==2.9.2
pydantic_core==2.23.4
pytest==8.3.4
pytest-asyncio==0.25.0
pytest-cov==6.0.0
requests==2.32.3
rsa==4.9
typing_extensions==4.12.2
urllib3==2.2.3
websockets==15.0.0



