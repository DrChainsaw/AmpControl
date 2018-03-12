package ampControl.audio.asio;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;

import com.beust.jcommander.Parameter;
import com.synthbot.jasiohost.AsioChannel;
import com.synthbot.jasiohost.AsioDriver;

import ampControl.audio.AudioInputBuffer;
import ampControl.audio.ClassifierInputProvider;
import ampControl.audio.ClassifierInputProviderFactory;
import ampControl.audio.Cnn2DInputProvider;
import ampControl.audio.processing.ProcessingResult;
import ampControl.audio.processing.SupplierFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link ClassifierInputProviderFactory} which creates input from a given channel on a given ASIO device.
 * 
 * @author Christian Skärby
 */
public class AsioClassifierInputFactory implements ClassifierInputProviderFactory {

    private static final Logger log = LoggerFactory.getLogger(AsioClassifierInputFactory.class);


    @Parameter(names = "-asioDriver", description = "Asio driver to use")
    private String driverName = "";

    @Parameter(names = "-channel", description = "Input channel to classify")
    private int channelInd = 0;

    // Should be final but can't thanks to JCommander...
    private AsioDriver driver;

    private final Map<AudioInputBuffer, Map<String, ClassifierInputProvider.Updatable>> inputProviderCache = new LinkedHashMap<>();
    private final Map<Integer, AudioInputBuffer> audioInputCache = new LinkedHashMap<>();
    private final Map<Integer, AsioChannel> activeAsioChannels = new HashMap<>();

    public void initialize() {
        // Effectively untestable as Asio stuff is not mockable
        AsioDriver.getDriverNames().forEach(name -> log.info("Found ASIO driver with name: " + name));
        driver = AsioDriver.getDriver(driverName);
        // No matter what, we must shut down the driver when program ends!
        setShutdownHook(driver);
    }

    private void setShutdownHook(AsioDriver driver) {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            log.info("Program ended, tear down driver");
            try{
                driver.stop();
            } finally {
                driver.shutdownAndUnloadDriver();
            }
        }));
    }

    @Override
    public ClassifierInputProvider createInputProvider(String inputDescriptionString) {
        AudioInputBuffer audioInput = createAudioBuffer(inputDescriptionString);
        Map<String, ClassifierInputProvider.Updatable> processingToInputProvider = inputProviderCache.get(audioInput);
        if(processingToInputProvider == null) {
            processingToInputProvider = new HashMap<>();
            inputProviderCache.put(audioInput, processingToInputProvider);
        }
        Supplier<ProcessingResult.Processing> resultSupplier = new SupplierFactory(driver.getSampleRate()).get(inputDescriptionString);
        ClassifierInputProvider.Updatable inputProvider = processingToInputProvider.get(resultSupplier.get().name());
        if(inputProvider == null) {
            inputProvider = new Cnn2DInputProvider(audioInput, resultSupplier);
            processingToInputProvider.put(resultSupplier.get().name(), inputProvider);
        }
        return inputProvider;
    }

    @Override
    public ClassifierInputProvider.UpdateHandle finalizeAndReturnUpdateHandle() {
        driver.createBuffers(new HashSet<>(activeAsioChannels.values()));
        driver.start();
        return () -> inputProviderCache.values().stream()
                .flatMap(procToInputMap -> procToInputMap.values().stream())
                .forEach(ClassifierInputProvider.UpdateHandle::updateInput);
    }

    private AudioInputBuffer createAudioBuffer(String str) {
        final int windowSizeMs = ClassifierInputProviderFactory.parseWindowSize(str);
        int nrofSamples = (int)(driver.getSampleRate() * windowSizeMs / 1000);
        final int key = hashKey(nrofSamples,channelInd);
        AudioInputBuffer inputBuffer = audioInputCache.get(key);
        if(inputBuffer == null) {

            AsioChannel channel = activeAsioChannels.get(channelInd);
            if(channel == null) {
                channel = driver.getChannelInput(channelInd);
                activeAsioChannels.put(channelInd, channel);
            }

            AsioAudioInputBuffer asioInputBuffer = new AsioAudioInputBuffer(
                    new SingleAsioInputChannel(channel),
                    driver.getBufferPreferredSize(),
                    nrofSamples);
            driver.addAsioDriverListener(asioInputBuffer);
            inputBuffer = asioInputBuffer;
            audioInputCache.put(key, inputBuffer);

        }
        return inputBuffer;
    }

    private static int hashKey(int... ints) {
        return Arrays.hashCode(ints);
    }
}
