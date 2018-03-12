# AmpControl

Do you have an amplifier hidden from plain sight with a dedicated wireless system for casual guitar practice in your living room? Are you sometimes annoyed that when playing along to recordings you can't find a sound which is a good balance for both rythm and leads but you also don't want to have a foot switch in your living room? Do you happen to have a moderately powerful computer standing close enough to the amplifier?

Didn't think so. But I had this very "problem" along with a desire to get my hands dirty with some machine learning so I created this software.

### What does it do?

In short: It changes the sound of the amplifier based on what you are playing on your guitar (or whatever instrument you plug in). 

This is done by continously feeding the contents of an audio buffer (e.g. from a soundcard) into a classifier (typically a neural network) which has been trained to classify short (50-200 ms) samples of audio. The output from the net is used to change the sound of the amplifier, e.g. change program, increase volume/gain/delay etc..

In order to be able to integrate the whole thing in a home automation system, the app can be controlled through a messaging protocol, e.g. MQTT.

The current interfaces are currently supported:

Audio input:
* ASIO (using [Jasiohost](https://github.com/mhroth/jasiohost))

Amp control output:
* MIDI (using javas native implementation)

App control:
* MQTT (using [Eclipse Paho Java](https://github.com/eclipse/paho.mqtt.java))    

Neural nets by [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)

### Does it really work?

Well, I would not replace my footswitch in live situations with this, but it works well enough in the casual practive context for me to be worth having. After a couple of more updates I might try it in rehearsals as well.

My best classifiers get about 96% accuracy on the test set which perhaps does not sound too bad. However, 96% accuracy means on average that you will get a "misclassification" every 10-40 seconds depending on how often the model produces a new classification. This is mitigated by using additional heuristics in the label probability interpretation, e.g filtering label probabilities over time, separate classification thresholds per label and not change sound unless you get a couple of classifications in a row saying the same. Using these techniques, I can mantain the error rate but it does cost a bit in delay so that switching does not feel "instant". 

Errors are obviously not random independently distributed but rather reflect cases which are hard for the model to distinguish. The hunt for the perfect model is continously ongoing though...

I do belive there are strict limiations with this approach. I have only attempted to classify into four different labels: Silence, Noise, Rythm, Lead. If one where to add e.g. clean sound in there I think the model would run into serious problems to distinguish between rythm, lead and clean given that it only sees a handful of milliseconds of audio at the time. Many guitarists (not me though) also use more than one sound for one type of playing and there is no guarantee that there exists some pattern only in the guitar input that tells which sound the artist playing had in mind, especially not one which would transfer between artists. Otoh, I'm no machine learning expert...

## Getting Started

Clone the repo and build with maven or just run from the IDE. The "main" applications are 
* ampControl.admin.AmpControlMain: Controls an amplifier as described above
* ampControl.model.training.TrainingDescription: Trains classifiers

I don't really expect anyone but me to use this project and as of now it does not contain any trained models nor the training data to create them as they are both quite sizey. Should you find this project and wish to try it out then apart from being very flattered I would also happily provide both the training data and any models. Just post it as an issue on the project, preferably with some recommended way of providing them. My dataset is 3.7 Gb and models are 20-100 Mb each. Data set consists of some of my music projects which happened to have consistent and clear track naming enough to automatically label the raw wav-files through a crude python script (not part of this repo as of now).


### Prerequisites

Maven, Git, Data set or models. Current configuration uses the CUDA 8.0 backend for the models and this requires CUDA 8.0 along with an NVIDIA GPU which supports CUDA 8.0. See https://deeplearning4j.org/gpu   

### Installing

Clone the repo and fire it up in your IDE of choice. I had some issues getting CUDA to work in eclipse due to some m2e bug/feature but this resolved when using IntelliJ. After I got it to work with IntelliJ it worked in eclipse for me as well.

## Running the tests

Unit tests are in the src/test folder. Some of the test cases instantiates models which might not fit on low tier GPUs (I have 6 Gb GPU RAM on my system). Please post an issue if you encounter any problems of this kind.

## Contributing

All contributions are very welcome. Please post issues to discuss any modification before creating a pull request. 

## Authors

* **Christian Skärby** - *Initial work* - [DrChainsaw](https://github.com/DrChainsaw)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j) - Neural networks for classifiers
* [Jcommander](http://jcommander.org/) - "Because life is too short to parse command line parameters"
* [Jasiohost](https://github.com/mhroth/jasiohost) - Audio input from ASIO sound cards
* [Eclipse Paho Java](https://github.com/eclipse/paho.mqtt.java) - For remote/automated control of the app
* [JTransforms](https://github.com/wendykierp/JTransforms) - FFTs and other transforms
* [jzy3d](http://www.jzy3d.org/) - 3D plotting of spectrograms and similar for debugging
* [xchart](https://github.com/timmolter/xchart) - Simple realtime plots 
* [Ganesh Tiwari](http://ganeshtiwaridotcomdotnp.blogspot.com) - "Borrowed" MFSC implementation
* [WinThing](https://github.com/msiedlarek/winthing) - "Borrowed" design of Engine from here
