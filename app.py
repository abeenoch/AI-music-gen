from transformers import pipeline
import scipy
import gradio as gr

# Load the text-to-audio model pipeline
synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

# Define a function to generate music based on user input
def generate_music(description):
    # Generate music using the text-to-audio pipeline
    music = synthesiser(description, forward_params={"do_sample": True})
    
    # Save the generated audio to a file
    output_file = "musicgen_out.wav"
    scipy.io.wavfile.write(output_file, rate=music["sampling_rate"], data=music["audio"])
    
    # Return the file path for Gradio to handle
    return output_file

# Create the Gradio interface
interface = gr.Interface(
    fn=generate_music,                         # The function to call
    inputs=gr.Textbox(label="Music Description"),  # Input widget
    outputs=gr.Audio(label="Generated Music"), # Output widget
    title="Music Generator",
    description="Enter a description of the type of music you'd like to generate. The AI will create music based on your input!"
)

# Launch the interface
interface.launch()
