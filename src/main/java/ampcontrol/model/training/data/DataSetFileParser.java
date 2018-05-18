package ampcontrol.model.training.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Utility for mapping all .wav files found under a directory (including subdirectories) to a {@link DataProviderBuilder}
 * based on a provided mapping. Uses hashcode for filenames to ensure that the same file will always be mapped to the
 * same {@link DataProviderBuilder} given the same mapping.
 * <br><br>
 * TODO: Where does this functionality really belong? Static method does not seem right
 * TODO: Simple directory for testing purposes.
 *
 * @author Christian Skärby
 *
 */
public class DataSetFileParser {
	
	private static final int MAX_NUM_WAVS_PER_CLASS = Integer.MAX_VALUE/40;
	
	public static void parseFileProperties(
			Path baseDir,
			Function<Double, DataProviderBuilder> dataSetMapper) throws IOException {

		List<String> labels = Files.walk(baseDir, 1)
				.filter(file -> !file.equals(baseDir))
				.filter(file -> Files.isDirectory(file))
				.map(file -> file.getFileName().toString())
				.sorted(String::compareToIgnoreCase)
				.collect(Collectors.toList());

		Files.walk(baseDir)
		.filter(file -> !Files.isDirectory(file))
		.filter(file -> file.getFileName().toString().matches(".*\\.wav"))
		.map(file -> dataSetMapper.apply(getSetIdentifier(file)).addFile(file))
		.sequential()
		.distinct()
		.forEach(builder -> labels.forEach(builder::addLabel));

	}
	
	private static double getSetIdentifier(Path file) {

		// Need to do some processing to prevent that very similar files end up in the same set
		// Firstly, ignore the directory name
		String hashName = file.getFileName().toString().replaceAll("_nohash_.*$","");
		// Then, assume files are called something like "track name_XX" where XX is take nr
		// Since hashcode seems to use the initial letters as MSBs reverse so that XX becomes the initial letters
		hashName = new StringBuilder(hashName).reverse().toString();
		//System.out.println(hashName);
        return (Math.abs(hashName.hashCode()) % 100);//(MAX_NUM_WAVS_PER_CLASS + 1)) *
      	//(100.0 / MAX_NUM_WAVS_PER_CLASS);
	}

	public static void main(String[] args) {
		System.out.println("Raw hash: " + ("E:\\Software projects\\python\\lead_rythm\\data\\lead\\dork triplets left_23".hashCode()) + " mod 100: "
				+ (Math.abs("E:\\Software projects\\python\\lead_rythm\\data\\lead\\dork triplets left_23".hashCode()) % (MAX_NUM_WAVS_PER_CLASS + 1)) *
				(100.0 / MAX_NUM_WAVS_PER_CLASS));

		System.out.println("Raw hash: " + ("E:\\Software projects\\python\\lead_rythm\\data\\lead\\dork triplets left_30".hashCode()) + " mod 100: "
				+ (Math.abs("E:\\Software projects\\python\\lead_rythm\\data\\lead\\dork triplets left_30".hashCode()) % (MAX_NUM_WAVS_PER_CLASS + 1)) *
		(100.0 / MAX_NUM_WAVS_PER_CLASS));

		System.out.println("Raw hash: " + ("23dork triplets left".hashCode()) + " mod 100: "
				+ (Math.abs("dork triplets left_23".hashCode()) % (MAX_NUM_WAVS_PER_CLASS + 1)) *
				(100.0 / MAX_NUM_WAVS_PER_CLASS));

		System.out.println("Raw hash: " + ("30dork triplets left".hashCode()) + " mod 100: "
				+ (Math.abs("30dork triplets left".hashCode()) % (MAX_NUM_WAVS_PER_CLASS + 1)) *
				(100.0 / MAX_NUM_WAVS_PER_CLASS));

		System.out.println("Raw hash: " + ("30chainsaw demo left".hashCode()) + " mod 100: "
				+ (Math.abs("30chainsaw demo left".hashCode()) % (MAX_NUM_WAVS_PER_CLASS + 1)) *
				(100.0 / MAX_NUM_WAVS_PER_CLASS));

		System.out.println(getSetIdentifier( Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\lead\\Chainsaw demo 1_14_nohash_0.wav")));
		System.out.println(getSetIdentifier( Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\lead\\Chainsaw demo 1_13_nohash_0.wav")));
		System.out.println(getSetIdentifier( Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\lead\\Chainsaw demo 1_12_nohash_0.wav")));
		System.out.println(getSetIdentifier( Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\lead\\Chainsaw demo 1_11_nohash_0.wav")));
		System.out.println(getSetIdentifier( Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\lead\\Chainsaw demo 1_76_nohash_4.wav")));
		System.out.println(getSetIdentifier( Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\lead\\Chainsaw demo 1_78_nohash_1.wav")));
		System.out.println(getSetIdentifier( Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\lead\\blog grogg fo 2_76_nohash_2.wav")));
		}
}
