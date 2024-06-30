import asyncio

from aiohttp import web


class AsyncCommandExecutor:
    def __init__(self):
        self.commands = asyncio.Queue()
        self.command_list = []  # Mirror list to keep track of commands
        self.current_process = None

    async def add_command(self, command):
        await self.commands.put(command)
        self.command_list.append(command)  # Add to the mirror list
        print(f"Added command: {command}")

    async def run_command(self, command):
        print(f"Executing: {command}")
        self.current_process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await self.current_process.communicate()
        print(f"Output: {stdout.decode()}")
        if stderr:
            print(f"Error: {stderr.decode()}")
        self.current_process = None  # Reset after completion
        self.command_list.remove(command)  # Remove from the mirror list when done

    async def run(self):
        while True:
            command = await self.commands.get()
            if command == "stop":
                break
            await self.run_command(command)

    async def interrupt_current_task(self):
        if self.current_process and not self.current_process.returncode:
            self.current_process.terminate()
            await self.current_process.wait()
            print("Current task interrupted and terminated.")
        else:
            print("No active process to terminate or process already completed.")

    def print_commands_in_queue(self):
        if self.command_list:
            print("Commands in the queue:")
            for command in self.command_list:
                print(command)
        else:
            print("No commands in the queue.")


async def add_command(request):
    data = await request.json()
    command = data.get("command")
    if command:
        await executor.add_command(command)
        return web.json_response({"status": "Command added successfully"})
    return web.json_response({"error": "No command provided"}, status=400)


async def interrupt_command(request):
    await executor.interrupt_current_task()
    return web.json_response({"status": "Current task interrupted and terminated"})


# Server setup and routes similar to previous examples


async def show_commands_in_queue(request):
    executor.print_commands_in_queue()
    return web.json_response({"status": "Printed commands in the queue"})


async def stop_server(request):
    await executor.add_command("stop")
    return web.json_response({"status": "Server stopping"})


executor = AsyncCommandExecutor()
app = web.Application()
app.add_routes(
    [
        web.post("/add", add_command),
        web.post("/interrupt", interrupt_command),
        web.get("/show", show_commands_in_queue),
        web.post("/stop", stop_server),
    ]
)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "localhost", 8080)
    loop.run_until_complete(site.start())
    loop.create_task(executor.run())
    loop.run_forever()
