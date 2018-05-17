package ampcontrol.admin;

import ampcontrol.admin.service.Service;
import ampcontrol.admin.service.classifiction.AudioClassificationService;
import ampcontrol.admin.service.control.AppControlService;
import ampcontrol.admin.service.control.ControlRegistry;
import com.beust.jcommander.Parameter;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.Collection;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Runs the program main loop. Design borrowed from Winthing project.
 *
 * Basically listens to messages from an {@link AppControlService} and either starts or stops the
 * {@link AudioClassificationService} or exits the application depending on which messages are received.
 *
 * @author Christian Skärby
 */
public class Engine {

	@Parameter(names = "-exitCmd", description = "Message to exit application")
	private String exitMsg = "exit";

	private static final Logger log = LoggerFactory.getLogger(Engine.class);
	
	private boolean isInit = false;
	private boolean exit = false;
	private AppControlService appControlService;
    private Collection<Service> services;

	private final Lock runnningLock = new ReentrantLock();
	private final Condition runningCondition = runnningLock.newCondition();
	private final Duration reconnectInterval = Duration.ofSeconds(5);

	/**
	 * Initialization. Reason for this method instead of a constructor is only because Jcommander must have
	 * an instance to set parameters.
	 *
	 * @param appControlService Controls the application
	 * @param services Services delivered by the application
	 */
	public void initialize(
			AppControlService appControlService,
			Collection<Service> services) {

		if (isInit) {
			throw new IllegalStateException("May not reinit!");
		}

		this.services = services;
		this.appControlService = appControlService;
		isInit = true;
	}

	/**
	 * Starts the main loop.
	 */
	public void run() {
		if (!isInit) {
			throw new IllegalStateException("Must initialize before running!");
		}
		exit = false;
		runnningLock.lock();
		try {
			while (!exit) {
				log.info("Starting engine...");
				boolean connected = false;
				try {
					final ControlRegistry registry = appControlService.start();
					registry.registerSubscription(exitMsg, this::stop);
					registry.setConnectionFailedAction(this::pokeCondition);
					services.forEach(service -> service.registerTo(registry));
					connected = true;
				} catch (final MqttException exception) {
					log.info("Could not connect: " + exception.getMessage() + "!");
				}
				if (connected) {
					try {
						log.info("Application started. Waiting for input.");
						runningCondition.await();
					} catch (final InterruptedException exception) {
						stop();                        
					}
				}

				if(!exit) {
					log.info(
							"Trying to reconnect in "+reconnectInterval.getSeconds() +" seconds..."
							);
					try {
						Thread.sleep(reconnectInterval.toMillis());
					} catch (final InterruptedException exception) {
						stop();
					}
				}
			}
		} finally {
			runnningLock.unlock();  
		}
		try {
			log.info("Engine done! " + exit);
			appControlService.stop();
		}  catch (final MqttException disconnectException) {
			log.info("Could not disconnect.");
			disconnectException.printStackTrace();
		}   
	}

	/**
	 * Stops the application.
	 */
	public void stop() {
		log.info("Stopping engine...");
		services.forEach(Service::stop);
		exit = true;
		pokeCondition();
	}
	
	private void pokeCondition() {
		runnningLock.lock();
		try {
			runningCondition.signal();
		} finally {
			runnningLock.unlock();
		}
	}
}

