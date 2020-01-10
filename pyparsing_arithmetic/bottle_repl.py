from bottle import default_app, route, request
from enum import Enum
import inspect
import threading
import random
from pprint import pprint
import io
# from markup import markupToHTML
from collections import deque, namedtuple
from datetime import datetime, timedelta
import textwrap
from arithmetic_parsing import BasicArithmeticParser
import cgitb
cgitb.enable()

FILE_NOT_FOUND_ERROR_RESPONSE = (404, "File not found")
OK_RESPONSE = 200
ONE_SECOND = timedelta(0, 1)
ONE_MINUTE = ONE_SECOND * 60
ONE_HOUR = ONE_MINUTE * 60
ONE_DAY = ONE_HOUR * 24

sessions_lock = threading.Lock()
sessions = {}
MAX_SESSIONS = 200
SESSION_KEY_LENGTH = 12
sessions_history = deque(maxlen=50)
CmdLog = namedtuple('CmdLog', 'timestamp player command')
cmd_history = deque(maxlen=50)

# value indexes into mutable PlayerStats.data
last_update = 0
last_cmd = 1
num_tests = 2
num_exceptions = 3
PlayerStats = namedtuple('PlayerStats', 'player game start_time data')


def str_to_datetime(s):
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def timedelta_to_str(td):
    secPerDay = 3600*24
    s = td.days*secPerDay + td.seconds
    m, s = divmod(s, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)
    else:
        return "%d:%02d" % (m, s)


def time_to_str(dt):
    return str(dt)[:-3]


def make_random_key(n=SESSION_KEY_LENGTH):
    return ''.join(random.choice("bcdfghjklmnpqrstvwxz0123456789") for _ in range(n))


def unique(seq):
    seen = set()
    for obj in seq:
        if obj not in seen:
            seen.add(obj)
            yield obj


class Repl:
    class CommandStatus(Enum):
        APP_SUCCESS = 0
        APP_FAILURE = 1
        META = 2
        META_QUIT = 3
        EMPTY_COMMAND = 99
    APP_COMMANDS = {CommandStatus.APP_SUCCESS, CommandStatus.APP_FAILURE}
    META_COMMANDS = {CommandStatus.META, CommandStatus.META_QUIT}
    meta_commands = "help examples vars clear code quit".split()

    def __init__(self, parser_class):
        self.parser = parser_class()

    def start_new_session(self):
        key = make_random_key()
        now = datetime.now()
        with sessions_lock:
            sessions[key] = PlayerStats(key, self, now, [now, '', 0, 0])
        return key

    def _update_session(self, key, cmdstr, success):
        with sessions_lock:
            session_stats = sessions[key].data
            session_stats[last_update] = datetime.now()
            if cmdstr:
                session_stats[last_cmd] = cmdstr
                if cmdstr.lower() not in self.meta_commands:
                    session_stats[num_tests] += 1
                    if not success:
                        session_stats[num_exceptions] += 1

    def _end_session(self, key):
        with sessions_lock:
            info = sessions.pop(key)
        sessions_history.append((key, info))

    def do_command(self, cmd, key):
        MAX_OUTPUT_LEN = 20000
        # print('cmd=', repr(cmd))
        if not cmd:
            return True, self.CommandStatus.EMPTY_COMMAND, ''
        elif cmd.lower() == "help":
            self._update_session(key, cmd, True)
            return True, self.CommandStatus.META, self.usage(self.parser)
        elif cmd.lower() == "examples":
            self._update_session(key, cmd, True)
            return True, self.CommandStatus.META, self.examples(self.parser)
        elif cmd.lower() == "vars":
            vars_str = io.StringIO()
            pprint(self.parser.vars(), stream=vars_str, width=30)
            self._update_session(key, cmd, True)
            return True, self.CommandStatus.META, vars_str.getvalue()
        elif cmd.lower() == "clear":
            self.parser = type(self.parser)()
            self._update_session(key, cmd, True)
            return True, self.CommandStatus.META, ''
        elif cmd.lower() == 'code':
            parser_class = type(self.parser)
            code = inspect.getsourcelines(parser_class)[0]
            code.append('\nparser = {}()\n'.format(parser_class.__name__))
            cmds = [hist.command for hist in cmd_history if hist.player == key
                    and hist.command.lower() not in self.meta_commands]
            code.append('print(parser.evaluate({!r}))\n'.format(cmds[-1] if cmds else 'area = π*r²'))
            self._update_session(key, cmd, True)
            return True, self.CommandStatus.META, ''.join(code)
        elif cmd.lower() == 'quit':
            self._update_session(key, cmd, True)
            self._end_session(key)
            return True, self.CommandStatus.META_QUIT, 'DONE'
        else:
            try:
                result = self.parser.evaluate(cmd)
                # print(cmd, result)
            except Exception as e:
                self._update_session(key, cmd, False)
                return False, self.CommandStatus.APP_FAILURE, "{}: {}".format(type(e).__name__, e)
            else:
                retvalue = repr(result)
                if len(retvalue) > MAX_OUTPUT_LEN:
                    retvalue = retvalue[:MAX_OUTPUT_LEN] + '...'
                if '\n' not in retvalue:
                    retvalue = '\n'.join(textwrap.wrap(retvalue, width=120))
                self._update_session(key, cmd, True)
                return True, self.CommandStatus.APP_SUCCESS, retvalue

    def get_last_command(self, key):
        with sessions_lock:
            session_info = sessions.get(key)
        if session_info is not None:
            return session_info.data[last_cmd]
        return ''

    @classmethod
    def make_name_list_string(cls, names, indent=''):
        import math
        import itertools
        names = list(unique(names))
        chunk_size = math.ceil(len(names) ** 0.5)
        chunks = [list(c) for c in itertools.zip_longest(*[iter(names)] * chunk_size, fillvalue='')]
        col_widths = [max(map(len, chunk)) for chunk in chunks]
        ret = []
        for transpose in zip(*chunks):
            line = indent
            for item, wid in zip(transpose, col_widths):
                line += "{:{}s}".format(item, wid + 2)
            ret.append(line.rstrip())
        return '\n'.join(ret)

    @classmethod
    def usage(cls, parser):
        msg = textwrap.dedent("""\
        Interactive utility to use the pyparsing-arithmetic BaseArithmeticParser.

        Use your keyboard or the buttons below to enter an arithmetic expression
        or assignment statement, using the following operators:
        {operator_list}

        Multiple assignments can be made using lists of variable names and
        corresponding lists of expressions (lists must be of matching lengths).
            x₁, y₁ = 1, 2
            a, b, c = 1, 2, a+b

        Deferred evaluation assignments can be defined using "@=":
            circle_area @= pi * r**2
        will re-evaluate 'circle_area' using the current value of 'r'.

        Expression can include the following functions:
        {function_list}

        Other commands:
        - vars - list all saved variable names
        - clear - clear saved variables
        - help - display this help text
        - quit - close your session (automatically starts a new one)
        - code - view sample code for creating and running this evaluator/parser
        """)
        func_list = cls.make_name_list_string(names=list({**parser.base_function_map, **parser.added_function_specs}),
                                              indent='  ')
        custom_operators = [str(oper_defn[0]) for oper_defn in parser.added_operator_specs]
        operators = parser.base_operators

        oper_list = cls.make_name_list_string(names=custom_operators + operators + ['|absolute-value|'], indent='  ')
        return msg.format(function_list=func_list, operator_list=oper_list)

    @classmethod
    def examples(cls, parser):
        msg = textwrap.dedent("""\
        Here are some example expressions that you can try, using the keyboard or included buttons:

            5!
            √2
            2√5
            15²
            12³
            sin(30)
            sin(rad(30))
            sin(30°)
            sin(pi/2)
            sin(π/2)
            sin(-π/2)
            |sin(-π/2)|
            1/0
            0**0
            3**2**3
            9**3
            3**8
            r = 100
            circle_area @= pi * r**2
            circle_area @= π×r²
            r = 20
            circle_area
            coin_toss @= rnd() > 0.5? "heads" : "tails"
            coin_toss
            play @= 'You ' + (rnd() > 0.5? 'win' : 'lose')
            play
            dice @= randint(1, 6)
            dice
            d3 @= randint(1, 6) + randint(1, 6) + randint(1, 6)
            1 or 0
            1 and not 0
            1 between 0 and 2
            100 between 0 and 100         (0 < 100 < 100    = False)
            100 within 0 and 100          (0 <= 100 <= 100  = True)
            100 in range from 0 to 100    (0 <= 100 < 100   = False)
            99.9 in range from 0 to 100
            dist @= ((x2-x1)**2 + (y2-y1)**2)**0.5
            dist @= √((x₂-x₁)² + (y₂-y₁)²)
            x₁, y₁ = 1, 2
            x₂, y₂ = 5, 6
            dist
        """)
        return msg


class BottleArithReplRequestHandler:

    def __init__(self):
        self.buffer = []

    def get_query(self):
        return {**request.query}

    def write_javascript(self, s):
        # print "JAVASCRIPT:", s
        self.buffer.append('\n<script language="JavaScript" type="text/javascript">\n')
        self.buffer.append(str(s) + '\n')
        self.buffer.append('</script>\n')

    def write_html(self, s):
        self.buffer.append(s)

    def write_html_text_block(self, s, fixed_font=False):
        if fixed_font:
            self.write_html("<pre>\n")
            self.write_html(s + "\n")
            self.write_html("</pre>\n")
        else:
            self.write_html(s.replace("\n", "<br>\n") + "<br>\n")

    def write_html_table(self, rowlist):
        buffer = []
        buffer.append("<table>")
        for row in rowlist:
            buffer.append("<tr><td>"+ "</td><td>".join(str(s) for s in row) + "</td></tr>")
        buffer.append("</table>")
        self.write_html('\n'.join(buffer))

    def _handle_app_request(self):
        title_string = "Pyparsing-based Arithmetic Parser/Evaluator Tester"
        # get any form input, pass to parser
        query = self.get_query()
        # print('query=', query)
        sessionkey = query.get('k')
        cmd = query.get('c', '').encode("latin1").decode('utf-8')
        if sessionkey is not None:
            cmd_history.append(CmdLog(datetime.now(), sessionkey, cmd))
        self.write_html('<html><head>\n')
        self.write_html('<meta name="HandheldFriendly" content="true" />')
        self.write_html('<meta name="viewport" content="width=device-width, initial-scale=1.0,'
                        ' maximum-scale=1.0, user-scalable=1" />')
        self.write_html('<meta charset="UTF-8">')
        self.write_html('<title>{}</title>'.format(title_string))
        self.write_html('</head>')
        self.write_html('<body OnLoad="document.turnForm.c.focus();">\n')
        self.write_html('<h3>{}</h3>\n<p>\n'.format(title_string))

        # get key from query
        game = None
        if sessionkey and sessionkey in sessions:
            with sessions_lock:
                session_info = sessions[sessionkey]
            player, game = session_info.player, session_info.game
            game_over = False
            success, command_status, output = game.do_command(cmd, sessionkey)
            # print(repr(cmd))
            # print(success)
            # print(output)
            if command_status is Repl.CommandStatus.META_QUIT:
                game_over = True

            if game_over:
                # self._sayRaw('<p>You ended the game with:\n<ul>\n')
                # for item in player.inv:
                #     self._sayRaw('<li>%s %s\n' % (advEng._aOrAn(str(item)), item))
                # self._sayRaw('</ul>\n')
                # if player.eventLog:
                #     self._sayRaw('<p><p>During the game you:\n<ul>\n')
                #     for evt in player.showLog():
                #         self._sayRaw('<li>%s\n' % evt)
                #     self._sayRaw('</ul>\n')
                pass
            else:
                if success or output:
                    self.write_html_text_block(cmd, fixed_font=True)
                    if isinstance(output, str):
                        self.write_html_text_block(output, fixed_font=True)
                    elif isinstance(output, list):
                        self.write_html_table(output)

        else:
            if len(sessions) < MAX_SESSIONS:
                # advEng.HelpCommand(None)._doCommand(None)
                # self._sayRaw(advEng.NLCHAR+'\n')
                # start new game
                game = Repl(BasicArithmeticParser)
                # create player session
                sessionkey = game.start_new_session()
            else:
                self.write_html('<h2>Sorry, too many sessions just now...</h2>\n')

        # print self.buffer
        # output latest game output

        # add buttons for operators and non-ASCII identifier chars
        def button(s, action='', go=False):
            if not action:
                action = s
            action = action.replace('"', r"&quot;").replace("`", r"\&grave;")
            flag = ('false', 'true')[go]
            action = "addtext(`" + action + "`," + flag + ");"
            self.write_html(
                '''<button onclick="{}">&nbsp;&nbsp;{}&nbsp;&nbsp;</button>\n'''.format(action, s)
                )

        def button_row(s, label=''):
            self.write_html('\n<br>{}\n'.format(label + ': ' if label else ''))
            for c in s:
                button(c)

        button_row('°√×÷≠≤≥∧∨', "Operators")
        button_row('+-*/=<>!()²³')
        # button_row('ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')
        # button_row('αβγδεζηθικλμνξοπρστυφχψω')
        # button_row('𝑎𝑏𝑐𝑑𝑒𝑓𝑔𝑖𝑗𝑘𝑙𝑚𝑛𝑜𝑝𝑞𝑟𝑠𝑡𝑢𝑣𝑤𝑥𝑦𝑧')
        button_row('abcdefghijklm')
        button_row('nopqrstuvwxyz')
        button_row('àáâãäåæçèéêëìíîï')
        button_row('ßðñòóôõöøùúûüýþÿ')
        button_row('αβγδεζηθικλμ')
        button_row('νξοπρστυφχψω')
        button_row('₀₁₂₃₄₅₆₇₈₉', "Subscripts")
        button_row("0123456789.πφ")
        self.write_html('\n<br>\n')
        if game is not None:
            previous_cmd = game.get_last_command(sessionkey)
            button('Prev', previous_cmd)
            self.write_html('&nbsp;&nbsp;')
            button('Redo', previous_cmd, go=True)
            self.write_html('&nbsp;&nbsp;')
        button('Help', 'help', go=True)
        button('Examples', 'examples', go=True)
        button('Vars', 'vars', go=True)
        button('Code', 'code', go=True)

        # output input field
        self.write_html('\n<br>\n<p>')
        self.write_html('\n<form name="turnForm"> Command: <input type="text" name="c" size="60">')
        self.write_html('\n<button name="go" type="submit">Submit</button>')
        # output hidden key
        self.write_html('\n<input type="hidden" name="k" value="KEY">'.replace("KEY", sessionkey))
        self.write_html('\n</form>')
        self.write_javascript(textwrap.dedent('''\
            function addtext(s, go) {
                var frm = document.turnForm;
                var input_field = frm.c;
                input_field.value += s;
                input_field.focus();
                input_field.setSelectionRange(10000,10000);
                if (go) {
                    frm.go.click();
                    }
                }'''))
        self.write_html('\n</body></html>')

    def _handle_stats_request(self):
        now = datetime.now()
        self.write_html('<html><body>\n')
        self.write_html('<h2>Stats as of {}</h2>\n<p>\n'.format(time_to_str(now)))
        headings = "Session/Start time/Latest time/Connected/Idle/Tests/Exceptions".split('/')
        self.write_html('<h2>Active Testers</h2>\n')
        self.write_html('<table border=1 cellpadding="4"><tr><th>' +
                        '</th><th>'.join(headings) +
                       '</th></tr>\n')
        if sessions:
            with sessions_lock:
                sessionsData = list(sessions.items())
            sessionsData.sort(key=lambda x:str(x[1].data[last_update]), reverse=True)
            for k, info in sessionsData:
                # explog = info.player.eventLog[:]
                # if explog:
                #     seen = set()
                #     exp = []
                #     for item in explog:
                #         if item in seen:
                #             continue
                #         times = explog.count(item)
                #         if times == 1:
                #             exp.append('%s' % item)
                #         else:
                #             exp.append('%s (%d times!)' % (item, times))
                #         seen.add(item)
                # else:
                #     exp = []

                connect_time = now - info.start_time
                connect_time_str = timedelta_to_str(connect_time)
                idle_time_str = timedelta_to_str(now - info.data[last_update])
                self.write_html('<tr><td valign="top">%s</td><td valign="top">%.19s</td><td valign="top">%.19s</td>'
                               '<td valign="top" align="center">%s</td>'
                               '<td valign="top" align="center">%s</td>'
                               '<td valign="top" align="center">%s</td>'
                               '<td valign="top" align="center">%s</td>\n' %
                                    (k, info.start_time, info.data[last_update],
                                     connect_time_str, idle_time_str, info.data[num_tests], info.data[num_exceptions]))
                # self.writeHTML('<tr><td valign="top">%s</td><td valign="top">%.19s</td><td valign="top">%.19s</td><td valign="top" width=200>%s</td><td valign="top" width=240>%s</td></tr>\n' %
                                    # (k, info.startTime, info.lastUpdate, playerInv, playerExp))
        else:
            self.write_html('<tr><td valign="top" colspan=%d><center>none</center></td></tr>\n' % len(headings))

        self.write_html('</table>\n')
        headings = "Session/Start time/End time/Connected/Tests/Exceptions/Commands".split('/')
        self.write_html('<p>\n')
        self.write_html('<h2>Finished Testers</h2>\n')
        self.write_html('<table border=1 cellpadding="4"><tr><th>' +
                        '</th><th>'.join(headings) +
                        '</th></tr>\n')
        sessions_data = list(sessions_history)
        if sessions_data:
            sessions_data.sort(key=lambda x:str(x[1].data[last_update]), reverse=True)
            for k, info in sessions_data:
                # explog = info.player.eventLog[:]
                # if explog:
                #     seen = set()
                #     exp = []
                #     for item in explog:
                #         if item in seen:
                #             continue
                #         times = explog.count(item)
                #         if times == 1:
                #             exp.append('%s' % item)
                #         else:
                #             exp.append('%s (%d times!)' % (item, times))
                #         seen.add(item)
                # else:
                #     exp = []
                connect_time = (info.data[last_update] - info.start_time)
                connect_time_str = timedelta_to_str(connect_time)
                self.write_html('<tr><td valign="top">%s</td><td valign="top">%.19s</td><td valign="top">%.19s</td>'
                               '<td valign="top" align="center">%s</td>'
                               '<td valign="top" align="center">%s</td>'
                               '<td valign="top" align="center">%s</td><td>%s</td>\n' %
                                    (k, info.start_time, info.data[last_update],
                                     connect_time_str, info.data[num_tests], info.data[num_exceptions], ''))
        else:
            self.write_html('<tr><td valign="top" colspan=%d><center>none</center></td></tr>\n' % len(headings))
        self.write_html('</table>\n')

        headings = "Time/Session/Command".split('/')
        self.write_html('<p>\n')
        self.write_html('<h2>Most Recent Commands</h2>\n')
        self.write_html('<table border=1 cellpadding="4"><tr><th>' +
                        '</th><th>'.join(headings) +
                       '</th></tr>\n')
        cmds = list(cmd_history)
        if cmds:
            cmds.sort(key=lambda x: x[0], reverse=True)
            for cmd in cmds:
                self.write_html('<tr><td valign="top">%.19s</td>'
                               '<td valign="top" align="center">%s</td>'
                               '<td valign="top">%s</td></tr>\n' % cmd)
        else:
            self.write_html('<tr><td valign="top" colspan=%d><center>none</center></td></tr>\n' % len(headings))
        self.write_html('</table>\n')
        self.write_html('\n</body></html>')

    def _handle_cleanup_request(self):
        with sessions_lock:
            sessions_data = list(sessions.items())

        deletes = []
        now = datetime.now()
        for k, info in sessions_data:
            idle_time = now - info.data[last_update]
            connect_time = info.data[last_update] - info.start_time
            if (connect_time < 2 * ONE_SECOND and idle_time > 5 * ONE_MINUTE
                    or idle_time > 4 * ONE_HOUR):
                deletes.append(k)

        with sessions_lock:
            for key in deletes:
                sessions.pop(key, None)

        return self._handle_stats_request()


@route('/arithrepl/_stats')
def handle_stats_command():
    handler = BottleArithReplRequestHandler()
    handler._handle_stats_request()
    return ''.join(handler.buffer)

@route('/arithrepl/_cleanup')
def handle_cleanup_command():
    handler = BottleArithReplRequestHandler()
    handler._handle_cleanup_request()
    return ''.join(handler.buffer)

@route('/arithrepl')
def handle_app_command():
    handler = BottleArithReplRequestHandler()
    handler._handle_app_request()
    return ''.join(handler.buffer)

@route('/')
def hello_world():
    return 'Hello from arith repl!'


application = default_app()
