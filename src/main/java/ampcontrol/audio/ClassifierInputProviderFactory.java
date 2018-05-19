package ampcontrol.audio;


import ampcontrol.audio.processing.ProcessingResult.Factory;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Factory interface for creating {@link ClassifierInputProvider ClassifierInputProviders}.
 *
 * @author Christian Skärby
 */
public interface ClassifierInputProviderFactory {
    Pattern timeWindowPattern = Pattern.compile(".*ws_(\\d*).*");


    /**
     * Creates a {@link ClassifierInputProvider} from the given string. String is typically expected to contain
     * information for what {@link Factory} shall be
     * applied and how large window size.
     *
     * @param inputDescriptionString description of what input the classifier expects
     * @return a {@link ClassifierInputProvider}
     */
    ClassifierInputProvider createInputProvider(String inputDescriptionString);

    /**
     * Finalizes input initialization (e.g. start driver) and return a {@link ClassifierInputProvider.UpdateHandle} for updating input.
     *
     * @return a {@link ClassifierInputProvider.UpdateHandle}
     */
    ClassifierInputProvider.UpdateHandle finalizeAndReturnUpdateHandle();

    /**
     * Generic utility for parsing out the time window size of a modelName
     * @param windowSizeString string containing window size
     * @return
     */
    static int parseWindowSize(String windowSizeString) {
        Matcher m = timeWindowPattern.matcher(windowSizeString);
        if(!m.find()) {
            throw new IllegalArgumentException("Could not find window size from " + windowSizeString);
        }
        return Integer.parseInt(m.group(1));
    }
}
