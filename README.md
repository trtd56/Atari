# about

This is program to solve the Atari problem of OpenAI Gym.
The method is Deep Q Neuralnetwork.

# Run command

~~~
$ python main.py -g {gpu number} -e {environment name}
~~~

- Breakout-v0

~~~
$ python main.py -g 0 -e Breakout-v0
~~~

- Pong-v0

~~~
$ python main.py -g 0 -e Pong-v0
~~~

# Reference

- If you see "OSError: [Errno 12] Cannot allocate memory.", try this command.

~~~
$ sudo bash -c "echo vm.overcommit_memory=1 >> /etc/sysctl.conf"
$ sudo sysctl -p
~~~

- If you're trying to render video on a server, you'll need to connect a fake display. 

~~~
$ xvfb-run -s "-screen 0 1400x900x24" bash
~~~

- If you not install ffmpeg, try this script.

~~~
$ sudo add-apt-repository ppa:mc3man/trusty-media
$ sudo apt-get update
$ sudo apt-get install ffmpeg
~~~
