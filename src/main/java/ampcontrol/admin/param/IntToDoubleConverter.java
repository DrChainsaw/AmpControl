package ampcontrol.admin.param;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.ParameterException;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Converts a comma separated string of colon separated pairs into a map.
 * <br>
 * Example: 0:1.2,3:4.5,6:7.8
 * @author Christian Skärby
 */
public class IntToDoubleConverter implements IStringConverter<Map<Integer, Double>> {


    @Override
    public Map<Integer, Double> convert(String s) {
        try {
            return Stream.of(s.split(","))
                    .map(str -> str.split(":"))
                    .collect(Collectors.toMap(
                            sPair -> Integer.parseInt(sPair[0]),
                            sPair -> Double.parseDouble(sPair[1]),
                            (a, b) -> a,
                            () -> new LinkedHashMap<>()
                    ));
        } catch (ArrayIndexOutOfBoundsException|NumberFormatException e) {
            throw new ParameterException("Incorrect formatting: " + s + "\n" + e);
        }
    }
}
