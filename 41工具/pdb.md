# PDB

[`pdb`](https://docs.python.org/zh-cn/3/library/pdb.html#module-pdb) 模块定义了一个交互式源代码调试器，用于 Python 程序。它支持在源码行间设置（有条件的）断点和单步执行，检视堆栈帧，列出源码列表，以及在任何堆栈帧的上下文中运行任意 Python 代码。它还支持事后调试，可以在程序控制下调用。

### 启动方式

#### 代码中断

中断进入调试器的典型用法是插入:

```
import pdb; pdb.set_trace()
```

或者:

```
breakpoint()
```

#### 命令行接口

你还可以从命令行唤起 [`pdb`](https://docs.python.org/zh-cn/3/library/pdb.html#module-pdb) 来调试其他脚本。 例如:

```
python -m pdb [-c command] (-m module | -p pid | pyfile) [args ...]
```

当作为模块被唤起时，如果被调试的程序异常退出则 pdb 将自动进入事后调试。 在事后调试之后（或程序正常退出之后），pdb 将重启程序。 自动重启会保留 pdb 的状态（如断点）并且在大多数情况下这比在退出程序的同时退出调试器更实用。

- **-c**, **--command** <command>

  要以在 `.pdbrc` 文件中所给出形式来执行命令；请参阅 [调试器命令](https://docs.python.org/zh-cn/3/library/pdb.html#debugger-commands)。*在 3.2 版本发生变更:* 增加了 `-c` 选项。

- **-m** <module>

  要以类似于 `python -m` 的方式来执行模块。 就像一个脚本那样，调试器将在模块的第一行之前暂停执行。*在 3.7 版本发生变更:* 增加了 `-m` 选项。

- **-p**, **--pid** <pid>

  附加到具有指定 PID 的进程。*在 3.14 版本加入.*

要附加到一个运行中的 Python 进程用于远程调试，请使用 `-p` 或 `--pid` 选项并传入目标进程的 PID:

```
python -m pdb -p 1234
```

备注

 

附加到一个在系统调用中被阻塞或等待I/O的进程上，只有当下一个字节码指令被执行或进程接收到一个信号时才会起作用。

在调试器控制下执行一条语句的典型用法如下:

```
import pdb
def f(x):
    print(1 / x)
pdb.run("f(2)")
> <string>(1)<module>()
(Pdb) continue
0.5
```

检查已崩溃程序的典型用法是:

```
import pdb
def f(x):
    print(1 / x)

f(0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in f
ZeroDivisionError: division by zero
pdb.pm()
> <stdin>(2)f()
(Pdb) p x
0
(Pdb)
```

### 命令

```
Documented commands (type help <topic>):
========================================
EOF    c          d        h         list      q        rv       undisplay
a      cl         debug    help      ll        quit     s        unt      
alias  clear      disable  ignore    longlist  r        source   until    
args   commands   display  interact  n         restart  step     up       
b      condition  down     j         next      return   tbreak   w        
break  cont       enable   jump      p         retval   u        whatis   
bt     continue   exit     l         pp        run      unalias  where 
```

#### 具体命令参考

- **h(elp)** [command]

  不带参数时，显示可用的命令列表。参数为 *command* 时，打印有关该命令的帮助。`help pdb` 显示完整文档（即 [`pdb`](https://docs.python.org/zh-cn/3/library/pdb.html#module-pdb) 模块的文档字符串）。由于 *command* 参数必须是标识符，因此要获取 `!` 的帮助必须输入 `help exec`。

- **w(here)** [count]

  打印堆栈跟踪，在底部显示最近的帧。如果 *count* 为0，则打印当前帧条目。如果 *count* 为负数，则打印最近的 *count* 帧。如果 *count* 为正数，打印最近的 *count* 帧。箭头（`>`）表示当前帧，它决定了大多数命令的上下文。*在 3.14 版本发生变更:* 增加了 *count* 参数。

- **d(own)** [count]

  在堆栈回溯中，将当前帧向下移动 *count* 级（默认为 1 级，移向更新的帧）。

- **u(p)** [count]

  在堆栈回溯中，将当前帧向上移动 *count* 级（默认为 1 级，移向更老的帧）。

- **b(reak)** [([filename:]lineno | function) [, condition]]

  传入 *lineno* 参数，在当前文件内的第 *lineno* 行设置中断。 行号数值开头可以带有 *filename* 加一个冒号，以在另一个文件内指定中断点（可能是尚未载入的文件）。 文件将根据 [`sys.path`](https://docs.python.org/zh-cn/3/library/sys.html#sys.path) 来搜索。 可接受的 *filename* 形式有 `/abspath/to/file.py`、 `relpath/file.py`、 `module` 和 `package.module`。传入 *function* 参数，在该函数内的第一条可执行语句上设置中断。 *function* 可以是会在当前命名空间中被求值为一个函数的任意表达式。如果第二个参数存在，它应该是一个表达式，且它的计算值为 true 时断点才起作用。如果不带参数执行，将列出所有中断，包括每个断点、命中该断点的次数、当前的忽略次数以及关联的条件（如果有）。每个中断点将被分配一个数值供所有其他中断点命令引用。

- **tbreak** [([filename:]lineno | function) [, condition]]

  临时断点，在第一次命中时会自动删除。它的参数与 [`break`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-break) 相同。

- **cl(ear)** [filename:lineno | bpnumber ...]

  如果参数是 *filename:lineno*，则清除此行上的所有断点。如果参数是空格分隔的断点编号列表，则清除这些断点。如果不带参数，则清除所有断点（但会先提示确认）。

- **disable** bpnumber [bpnumber ...]

  禁用断点，断点以空格分隔的断点编号列表给出。禁用断点表示它不会导致程序停止执行，但是与清除断点不同，禁用的断点将保留在断点列表中并且可以（重新）启用。

- **enable** bpnumber [bpnumber ...]

  启用指定的断点。

- **ignore** bpnumber [count]

  为指定的断点编号设置忽略次数。 如果省略 *count*，则忽略次数将设置为 0。 当忽略次数为零时断点将变为活动状态。 如果为非零值，则在每次到达断点且断点未禁用且关联条件取真值时 *count* 就像递减。

- **condition** bpnumber [condition]

  为断点设置一个新 *condition*，它是一个表达式，且它的计算值为 true 时断点才起作用。如果没有给出 *condition*，则删除现有条件，也就是将断点设为无条件。

- **commands** [bpnumber]

  为编号是 *bpnumber* 的断点指定一系列命令。命令内容将显示在后续的几行中。输入仅包含 `end` 的行来结束命令列表。举个例子:`(Pdb) commands 1 (com) p some_variable (com) end (Pdb) `要删除断点上的所有命令，请输入 `commands` 并立即以 `end` 结尾，也就是不指定任何命令。如果不带 *bpnumber* 参数，`commands` 作用于最后一个被设置的断点。可以为断点指定命令来重新启动程序。只需使用 [`continue`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-continue) 或 [`step`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-step) 命令或其他可以继续运行程序的命令。如果指定了某个继续运行程序的命令（目前包括 [`continue`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-continue)、 [`step`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-step)、 [`next`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-next)、 [`return`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-return)、[`until`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-until)、 [`jump`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-jump)、 [`quit`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-quit) 及它们的缩写）将终止命令列表（就像该命令后紧跟着 end）。因为在任何时候继续运行下去（即使是简单的 next 或 step），都可能会遇到另一个断点，该断点可能具有自己的命令列表，这导致要执行的列表含糊不清。如果命令列表中包含``silent``命令或恢复执行的命令，则不显示包含帧信息的断点消息。*在 3.14 版本发生变更:* 如果命令列表中存在恢复执行的命令，则不会显示帧信息。

- **s(tep)**

  运行当前行，在第一个可以停止的位置（在被调用的函数内部或在当前函数的下一行）停下。

- **n(ext)**

  继续运行，直到运行到当前函数的下一行，或当前函数返回为止。（ [`next`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-next) 和 [`step`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-step) 之间的区别在于，[`step`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-step) 进入被调用函数内部并停止，而 [`next`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-next) （几乎）全速运行被调用函数，仅在当前函数的下一行停止。）

- **unt(il)** [lineno]

  如果不带参数，则继续运行，直到行号比当前行大时停止。如果带有 *lineno*，则继续执行直至行号大于或等于 *lineno*。 在这两种情况下，在当前帧返回时也将停止。*在 3.2 版本发生变更:* 允许明确给定行号。

- **r(eturn)**

  继续运行，直到当前函数返回。

- **c(ont(inue))**

  继续运行，仅在遇到断点时停止。

- **j(ump)** lineno

  设置即将运行的下一行。仅可用于堆栈最底部的帧。它可以往回跳来再次运行代码，也可以往前跳来跳过不想运行的代码。需要注意的是，不是所有的跳转都是允许的 -- 例如，不能跳转到 [`for`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#for) 循环的中间或跳出 [`finally`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#finally) 子句。

- **l(ist)** [first[, last]]

  列出当前文件的源代码。如果不带参数，则列出当前行周围的 11 行，或延续前一次列出。如果用 `.` 作为参数，则列出当前行周围的 11 行。如果带有一个参数，则列出那一行周围的 11 行。如果带有两个参数，则列出所给的范围中的代码；如果第二个参数小于第一个参数，则将其解释为列出行数的计数。当前帧中的当前行用 `->` 标记。如果正在调试异常，且最早抛出或传递该异常的行不是当前行，则那一行用 `>>` 标记。*在 3.2 版本发生变更:* 增加了 `>>` 标记。

- **ll** | longlist

  列出当前函数或帧的所有源代码。相关行的标记与 [`list`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-list) 相同。*在 3.2 版本加入.*

- **a(rgs)**

  打印当前函数的参数及其当前的值。

- **p** expression

  在当前上下文中对 *expression* 求值并打印该值。备注 `print()` 也可以使用，但它不是一个调试器命令 --- 它执行 Python [`print()`](https://docs.python.org/zh-cn/3/library/functions.html#print) 函数。

- **pp** expression

  与 [`p`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-p) 命令类似，但 *expression* 的值将使用 [`pprint`](https://docs.python.org/zh-cn/3/library/pprint.html#module-pprint) 模块美观地打印。

- **whatis** expression

  打印 *expression* 的类型。

- **source** expression

  尝试获取 *expression* 的源代码并显示它。*在 3.2 版本加入.*

- **display** [expression]

  如果 *expression* 的值发生变化则显示它的值，每次都会停止执行当前帧。如果不带 *expression*，则列出当前帧的所有显示表达式。备注 显示 *expression* 的值并与 *expression* 之前的求值结果进行比较，因此当结果可变时，显示可能无法体现变化。示例：`lst = [] breakpoint() pass lst.append(1) print(lst) `显示将不会发现 `lst` 已被改变因为求值结果在执行比较之前已被 `lst.append(1)` 原地修改了:`> example.py(3)<module>() -> pass (Pdb) display lst display lst: [] (Pdb) n > example.py(4)<module>() -> lst.append(1) (Pdb) n > example.py(5)<module>() -> print(lst) (Pdb) `你可以通过拷贝机制巧妙地实现此功能:`> example.py(3)<module>() -> pass (Pdb) display lst[:] display lst[:]: [] (Pdb) n > example.py(4)<module>() -> lst.append(1) (Pdb) n > example.py(5)<module>() -> print(lst) display lst[:]: [1]  [old: []] (Pdb) `*在 3.2 版本加入.*

- **undisplay** [expression]

  不再显示当前帧中的 *expression*。 如果不带 *expression*，则清除当前帧的所有显示表达式。*在 3.2 版本加入.*

- **interact**

  在根据当前作用域的局部和全局命名空间初始化的新建全局命名空间中启动一个交互式解释器 (使用 [`code`](https://docs.python.org/zh-cn/3/library/code.html#module-code) 模块)。 使用 `exit()` 或 `quit()` 退出解释器并返回调试器。备注 由于 `interact` 为代码执行创建了一个新的专用命名空间，对变量的赋值将不会影响原始命名空间。 不过，对任何被引用的可变对象的修改都将照常在原始命名空间中反映出来。*在 3.2 版本加入.**在 3.13 版本发生变更:* 可以使用 `exit()` 和 `quit()` 退出 [`interact`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-interact) 命令。*在 3.13 版本发生变更:* [`interact`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-interact) 会将其输出引向调试器的输出通道而不是 [`sys.stderr`](https://docs.python.org/zh-cn/3/library/sys.html#sys.stderr)。

- **alias** [name [command]]

  创建一个名为 *name* 的别名用来执行 *command*。 *command* 的两边 *不可* 用引号括起来。 可替换形参可以通过 `%1`, `%2` ... 和 `%9` 等来指明，而 `%*` 将被所有这些形参替换。 如果省略 *command*，则将显示 *name* 的当前别名。 如未给出任何参数，则将列出所有别名。别名允许嵌套并可包含能在 pdb 提示符下合法输入的任何内容。 请注意内部 pdb 命令 *可以* 被别名所覆盖。 这样的命令将被隐藏直到别名被移除。 别名会递归地应用到命令行的第一个单词；行内的其他单词不会受影响。作为示例，这里列出了两个有用的别名（特别适合放在 `.pdbrc` 文件中）:`# 打印实例变量 (用法 "pi classInst") alias pi for k in %1.__dict__.keys(): print(f"%1.{k} = {%1.__dict__[k]}") # 打印 self 中的实例变量 alias ps pi self `

- **unalias** name

  删除指定的别名 *name*。

- **!** statement

  在当前栈帧的上下文中执行 (单行的) *statement*。 感叹号可以被省略，除非第一个语句的第一个单词与某个调试器命名重名，例如:`(Pdb) ! n=42 (Pdb) `要设置全局变量，你可以在同一行上在赋值命令前添加 [`global`](https://docs.python.org/zh-cn/3/reference/simple_stmts.html#global) 语句，例如:`(Pdb) global list_options; list_options = ['-l'] (Pdb) `

- **run** [args ...]

- **restart** [args ...]

  重启被调试的 Python 程序。 如果提供了 *args*，它会用 [`shlex`](https://docs.python.org/zh-cn/3/library/shlex.html#module-shlex) 来拆分且拆分结果将被用作新的 [`sys.argv`](https://docs.python.org/zh-cn/3/library/sys.html#sys.argv)。 历史、中断点、动作和调试器选项将被保留。 [`restart`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-restart) 是 [`run`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-run) 的一个别名。*在 3.14 版本发生变更:* [`run`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-run) 和 [`restart`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-restart) 命令在 `'inline'` 模式下调用调试器时被禁用。

- **q(uit)**

  退出调试器。 被执行的程序将被中止。输入文件结束符相当于 [`quit`](https://docs.python.org/zh-cn/3/library/pdb.html#pdbcommand-quit)。如果调试器以 `'inline'` 模式调用，将显示一个确认提示符。`y`、`Y`、`<Enter>` 或 `EOF` 都将确认退出。*在 3.14 版本发生变更:* 如果调试器以 `'inline'` 模式调用，将显示一个确认提示符。 确认后，调试器将立即调用 [`sys.exit()`](https://docs.python.org/zh-cn/3/library/sys.html#sys.exit)，而不是在下一个跟踪事件中引发 [`bdb.BdbQuit`](https://docs.python.org/zh-cn/3/library/bdb.html#bdb.BdbQuit)。

- **debug** code

  进入一个对 *code* 执行步进的递归调试器（该参数是在当前环境中执行的任意表达式或语句）。

- **retval**

  打印当前函数最后一次返回的返回值。

- **exceptions** [excnumber]

  列出串连的异常或在其间跳转。当将 `pdb.pm()` 或 `Pdb.post_mortem(...)` 用于串连的异常而不是回溯数据时，它允许用户使用 `exceptions` 命令在串连的异常间移动以列出异常，并使用 `exceptions <number>` 来切换到该异常。示例：`def out():    try:        middle()    except Exception as e:        raise ValueError("reraise middle() error") from e def middle():    try:        return inner(0)    except Exception as e:        raise ValueError("Middle fail") def inner(x):    1 / x  out() `调用 `pdb.pm()` 将允许在异常之间移动:`> example.py(5)out() -> raise ValueError("reraise middle() error") from e (Pdb) exceptions  0 ZeroDivisionError('division by zero')  1 ValueError('Middle fail') > 2 ValueError('reraise middle() error') (Pdb) exceptions 0 > example.py(16)inner() -> 1 / x (Pdb) up > example.py(10)middle() -> return inner(0)`

### 参考

1. [pdb --- Python 的调试器](https://docs.python.org/zh-cn/3/library/pdb.html)

