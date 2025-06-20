ACTION_HINTS = """
click(x: float, y: float, button: Literal['left', 'right'] = 'left', double: bool = False)
    Move the mouse to a location and click a mouse button.
    Can be used to click a button, select a checkbox, focus on a input field, etc.
    Args:
        x (float): The x coordinate of the location to click.
        y (float): The y coordinate of the location to click.
        button (Literal["left", "right"]): The button to click.
        double (bool): Whether to double click.
    Examples:
        click(324.5, 12)
        click(119, 34, button="right")
        click(34.1, 720, double=True)
        click(230, 100, button="left", double=False)

complete(answer: str = '', infeasible_reason: str = '')
    Complete the task and optionally provide the user some feedback.
    Fill in the answer if the completion of the task requires providing a response to the user.
    Fill in the infeasible_reason if the task is infeasible.
    DO NOT fill in both answer and infeasible_reason at the same time.
    Args:
        answer (str): The answer to the task, if any.
        infeasible_reason (str): The reason the task is infeasible, if any.
    Examples:
        complete(answer="To request a refund, you need to call the customer service at 123-456-7890.")
        complete(infeasible_reason="The task is infeasible because the user has not provided a valid email address.")
        complete()
        complete(answer="{\n  "name": "John",\n  "age": 30,\n  "city": "New York"\n}")

drag_and_release(x1: float, y1: float, x2: float, y2: float)
    Press down the left mouse button at a location, drag the mouse to another location, and release the mouse button.
    Can be used for selecting a section of text, dragging a slider, etc.
    Args:
        x1 (float): The x coordinate of the location to press down the left mouse button.
        y1 (float): The y coordinate of the location to press down the left mouse button.
        x2 (float): The x coordinate of the location to release the left mouse button.
        y2 (float): The y coordinate of the location to release the left mouse button.
    Examples:
        drag_and_release(10.5, 200, 10.5, 230)

hover(x: float, y: float)
    Move the mouse to a location and stay there.
    Can be used to focus on a location, pop up a tooltip, navigate a dropdown menu, etc.
    Args:
        x (float): The x coordinate of the location to hover over.
        y (float): The y coordinate of the location to hover over.
    Examples:
        hover(102, 720)

key_press(keys: list[str])
    Press one or a combination of keys at the same time on the keyboard.
    Can be used
    - As various shortcuts of the current environment (e.g. ["Control", "a"], ["Control", "f"]).
    - To complete a search with ["Enter"].
    - And any other common actions that can be performed with a keyboard in the relevant application.
    This should NOT be used to type a string of text. Use the type action for that.
    The list of allowed keys are:
    - F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
    - 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    - a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
    - Backspace, Tab, Enter, Shift, Control, Alt, Delete
    - ArrowUp, ArrowDown, ArrowLeft, ArrowRight
    - Home, End, PageUp, PageDown
    Args:
        keys (list[str]): The list of keys to press.
    Examples:
        key_press(["Control", "a"]) # Select all
        key_press(["Control", "f"]) # Open the search bar
        key_press(["Enter"]) # Submit a form
        key_press(["F12"]) # Open the developer tools in a browser

scroll(x: float, y: float, delta_x: float = 0, delta_y: float = 100)
    Move the mouse to a location and scroll the mouse wheel in the x and y directions.
    Can be used to scroll a webpage, scroll a dropdown menu, etc.
    Args:
        x (float): The x coordinate of the location to scroll over.
        y (float): The y coordinate of the location to scroll over.
        delta_x (float): The amount to scroll horizontally.
        delta_y (float): The amount to scroll vertically.
    Examples:
        scroll(102, 320)
        scroll(102, 320, 0, 200)
        scroll(90, 32.7, 0, -300)
        scroll(620, 105, 68, 49.6)

type(x: float, y: float, text: str)
    Focus on a location and type a string of text in it.
    Can be used to type in a text field, search bar, edit a document, etc.
    Args:
        x (float): The x coordinate of the location to type text in.
        y (float): The y coordinate of the location to type text in.
        text (str): The text to type.
    Examples:
        type(102, 70.6, "\nThank you for the coffee!\n")
        type(44, 120, "Best sellers")

wait(ms: int = 1000)
    Wait for a specified amount of time.
    Can be used to wait for a webpage to load, a long form to display, etc.
    Args:
        ms (int): The amount of time to wait in milliseconds.
    Examples:
        wait()
        wait(1000)
        wait(500)
"""

# SFT Prompt Templates
SFT_SYSTEM_PROMPT = """You are a powerful and precise web agent, helping a user with their web-related tasks. 
Here is the set of actions you can take on a webpage, which you can call as python functions. 
It is a documentation of all the functions, and it's important that you read it well and carefully: 
{action_hints}
The user will provide the screenshot of the webpage, and you will need to use the actions to complete the task. 
Your outputs should be single python function calls, without any additional text, and should follow the correct formatting given above. 
Refer to the documentation to determine appropriate args. 
DO NOT PROVIDE MORE THAN ONE FUNCTION CALL AT EACH TURN! 
You will also be given the history of your previous thoughts and actions, use this to correct your trajectory intelligently."""

SFT_USER_PROMPT = """Please help me! 
We are trying to complete the following tasks: {instruction}
If a previous action failed or has been repeated multiple times without a positive outcome, please avoid repeating the same mistakes. 
Try to use a completely different approach. 
The previous action can fail either because the action itself is inappropriate, or because the coordinates of the elements are not correct. 
Adjust your next action accordingly. 
Here is the current screenshot of the webpage, which you can interact with using the actions and which you should remember coordinates for elements: 
<image>
Pixel coordinates originate from the top left corner of the image, where the first coordinate refers to the horizontal/width axis and the second refers to the vertical/height axis. 
Important: explore, explore, explore! The screenshot is not the entire webpage and you need to scroll to determine if a task is completable or whether you have gathered all the information you need. 
Again, our goal is: {instruction}
"""

def get_sft_messages(instruction: str, center_x: float, center_y: float):
    """Generate SFT format messages for training."""
    system_content = SFT_SYSTEM_PROMPT.format(action_hints=ACTION_HINTS)
    user_content = SFT_USER_PROMPT.format(instruction=instruction)

    if center_x is not None and center_y is not None:
        assistant_content = f"<answer>click({center_x:.0f}, {center_y:.0f})</answer>"
    else:
        assistant_content = ""
    
    prompt = [
        {"role": "user", "content": system_content}, # This is NOT a typo.. This is done so that qwen adds it's own default system prompt on top of this
        {"role": "user", "content": user_content}
    ]
    
    response = [
        {"role": "assistant", "content": assistant_content}
    ]
    
    return prompt, response
