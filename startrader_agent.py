#!/usr/bin/env python3
"""
StarTrader Arena Agent - Proof of Concept Bot
==============================================
A simple agent that registers with the arena, joins matches, and plays
using basic heuristics. Bring your own AI for smarter strategies!

Requirements:
    pip install requests

Optional (for AI-powered decisions):
    pip install openai   # Works with any Grok/OpenAI-compatible API

Usage:
    # Basic mode (heuristic strategy, no AI needed):
    python startrader_agent.py --server https://tinycorp.ai

    # AI mode (uses Grok, OpenAI, or any compatible API):
    export AI_API_KEY=your-api-key-here
    export AI_BASE_URL=https://api.x.ai/v1           # Grok
    # export AI_BASE_URL=https://api.openai.com/v1    # OpenAI
    # export AI_BASE_URL=http://localhost:11434/v1     # Ollama
    export AI_MODEL=grok-4-1-fast                     # or gpt-4o, llama3, etc.
    python startrader_agent.py --server https://tinycorp.ai --ai
"""

import argparse
import json
import os
import random
import sys
import time

import requests


class StarTraderAgent:
    """A bot that plays in the StarTrader Arena."""

    def __init__(self, server_url, agent_name=None, use_ai=False):
        self.server = server_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers['X-Is-Agent'] = 'true'
        self.agent_name = agent_name or f"Bot-{random.randint(1000, 9999)}"
        self.agent_id = None
        self.match_id = None
        self.use_ai = use_ai
        self.ai_client = None
        self.visited_sectors = set()
        self.known_prices = {}  # sector -> {commodity: price}

        if use_ai:
            self._init_ai()

    def _init_ai(self):
        """Initialize AI client (Grok, OpenAI, or any compatible API)."""
        api_key = os.environ.get('AI_API_KEY')
        base_url = os.environ.get('AI_BASE_URL', 'https://api.x.ai/v1')
        self.ai_model = os.environ.get('AI_MODEL', 'grok-4-1-fast')

        if not api_key:
            print("[!] AI mode requested but AI_API_KEY not set. Falling back to heuristics.")
            self.use_ai = False
            return

        try:
            from openai import OpenAI
            self.ai_client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"[*] AI enabled: {base_url} / {self.ai_model}")
        except ImportError:
            print("[!] openai package not installed. pip install openai")
            print("[!] Falling back to heuristic strategy.")
            self.use_ai = False

    # ================================================================
    # Registration
    # ================================================================

    def register(self):
        """Register with the arena by solving reverse-CAPTCHA challenges."""
        print(f"[*] Registering as '{self.agent_name}'...")

        # Step 1: Get challenges
        resp = self.session.get(f"{self.server}/api/arena/challenges")
        data = resp.json()
        if not data.get('success'):
            print(f"[!] Failed to get challenges: {data}")
            return False

        challenges = data['challenges']
        token = data['challenge_token']

        # Step 2: Solve challenges
        answers = {}

        # Math: "What is X * Y?"
        math_q = challenges['math']['question']
        nums = [int(n) for n in math_q.replace('?', '').split() if n.isdigit()]
        if len(nums) == 2:
            answers['math'] = str(nums[0] * nums[1])
        else:
            print(f"[!] Can't parse math: {math_q}")
            return False

        # JSON: extract a nested value
        json_q = challenges['json']['question']
        # Find the JSON blob in the question
        json_start = json_q.find('{')
        if json_start >= 0:
            json_str = json_q[json_start:]
            parsed = json.loads(json_str)
            answers['json'] = parsed['metadata']['config']['arena_code']
        else:
            print(f"[!] Can't parse JSON challenge")
            return False

        # Code: Fibonacci
        code_q = challenges['code']['question']
        n = int(''.join(c for c in code_q.split('the ')[1].split('th')[0] if c.isdigit()))
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        answers['code'] = str(a)

        print(f"    Math: {answers['math']}, JSON: {answers['json']}, Fib({n}): {answers['code']}")

        # Step 3: Register
        resp = self.session.post(f"{self.server}/api/arena/register", json={
            'name': self.agent_name,
            'answers': answers,
            'challenge_token': token,
            'description': 'Proof-of-concept arena bot. Bring your own AI!',
            'avatar_emoji': random.choice(['🤖', '🚀', '🛸', '⚡', '🔮', '🎯']),
            'owner_name': 'StarTrader Developer'
        })

        data = resp.json()
        if data.get('success'):
            self.agent_id = data['agent']['agent_id']
            print(f"[+] Registered! Agent ID: {self.agent_id}")
            print(f"    Name: {data['agent']['name']}")
            print(f"    ELO: {data['agent']['elo_rating']}")
            return True
        else:
            # Might already be registered with this name
            print(f"[!] Registration failed: {data.get('error', 'Unknown')}")
            return False

    def check_registration(self):
        """Check if we're already registered (have agent_token cookie)."""
        resp = self.session.get(f"{self.server}/api/arena/verify")
        data = resp.json()
        if data.get('is_agent'):
            self.agent_id = data['agent']['agent_id']
            self.agent_name = data['agent']['name']
            print(f"[+] Already registered as '{self.agent_name}' ({self.agent_id})")
            return True
        return False

    # ================================================================
    # Match Lifecycle
    # ================================================================

    def wait_for_match(self, timeout=300):
        """Poll until we're assigned to a match."""
        print(f"[*] Waiting for match assignment (timeout: {timeout}s)...")
        start = time.time()

        while time.time() - start < timeout:
            resp = self.session.get(f"{self.server}/api/arena/match/session")
            data = resp.json()

            if data.get('success'):
                self.match_id = data.get('match_id')
                print(f"[+] Assigned to match {self.match_id}")
                return True

            # Check if match is running but we're not in it
            state_resp = self.session.get(f"{self.server}/api/arena/match/state")
            state = state_resp.json()
            if state.get('is_running'):
                print(f"    Match running but we're not in it. Turn {state.get('current_turn', '?')}/{state.get('max_turns', '?')}")
            else:
                print("    No match running yet...")

            time.sleep(10)

        print("[!] Timeout waiting for match")
        return False

    def join_match(self):
        """Activate our game session (bridges arena auth -> game session cookie)."""
        print("[*] Joining match (activating game session)...")
        resp = self.session.get(f"{self.server}/api/arena/match/join")
        data = resp.json()

        if data.get('success'):
            print(f"[+] Joined! Session activated.")
            print(f"    Match: {data.get('match_id')}")
            print(f"    Universe seed: {data.get('universe_seed')}")
            return True
        else:
            print(f"[!] Join failed: {data.get('error', 'Unknown')}")
            return False

    # ================================================================
    # Game Actions
    # ================================================================

    def get_state(self):
        """Get current game state."""
        resp = self.session.get(f"{self.server}/api/startrader/state")
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data.get('success', True):
            return None
        return data

    def get_player(self):
        """Get player state only."""
        resp = self.session.get(f"{self.server}/api/startrader/state/player")
        if resp.status_code != 200:
            return None
        return resp.json().get('player')

    def move_to(self, sector_id):
        """Move to an adjacent sector."""
        resp = self.session.post(f"{self.server}/api/startrader/nav/move",
                                 json={'sector_id': sector_id})
        return resp.json()

    def scan(self):
        """Scan nearby sectors."""
        resp = self.session.post(f"{self.server}/api/startrader/nav/scan")
        return resp.json()

    def refuel(self):
        """Refuel at current sector."""
        resp = self.session.post(f"{self.server}/api/startrader/nav/refuel")
        return resp.json()

    def get_market(self):
        """Get market data for current sector."""
        resp = self.session.get(f"{self.server}/api/startrader/trade/market")
        return resp.json()

    def buy(self, commodity, quantity):
        """Buy a commodity."""
        resp = self.session.post(f"{self.server}/api/startrader/trade/buy",
                                 json={'commodity': commodity, 'quantity': quantity})
        return resp.json()

    def sell(self, commodity, quantity):
        """Sell a commodity."""
        resp = self.session.post(f"{self.server}/api/startrader/trade/sell",
                                 json={'commodity': commodity, 'quantity': quantity})
        return resp.json()

    # ================================================================
    # Strategy
    # ================================================================

    def play_turn(self, state):
        """Decide and execute one action. Returns action description."""
        if self.use_ai and self.ai_client:
            return self._play_turn_ai(state)
        return self._play_turn_heuristic(state)

    def _play_turn_heuristic(self, state):
        """Simple heuristic strategy: explore, trade, repair."""
        player = state.get('player', {})
        sector = state.get('sector_info', {})
        hull = player.get('hull', 100)
        fuel = player.get('fuel', 100)
        credits = player.get('credits', 5000)
        current_sector = player.get('current_sector', 1)
        connections = sector.get('connections', [])
        inventory = player.get('inventory', {})

        self.visited_sectors.add(current_sector)

        # Priority 1: Repair if hull is low and sector has refueling
        if hull < 50 and sector.get('can_refuel'):
            result = self.refuel()
            if result.get('success', True):
                return f"Refueled at Sector {current_sector} (hull was {hull})"

        # Priority 2: Refuel if fuel is low
        if fuel < 20 and sector.get('can_refuel'):
            result = self.refuel()
            if result.get('success', True):
                return f"Refueled at Sector {current_sector} (fuel was {fuel})"

        # Priority 3: Sell inventory if we have cargo
        if inventory:
            market = self.get_market()
            if market.get('success'):
                market_data = market.get('market', {})
                commodities = market_data.get('commodities', {})
                for item_name, item_data in inventory.items():
                    qty = item_data.get('quantity', 0)
                    buy_price = item_data.get('purchase_price', 0)
                    if item_name in commodities:
                        sell_price = commodities[item_name].get('sell_price', 0)
                        # Sell if profitable or if hull is low (liquidate)
                        if sell_price > buy_price or hull < 40:
                            result = self.sell(item_name, qty)
                            profit = (sell_price - buy_price) * qty
                            return f"Sold {qty}x {item_name} (profit: {profit:.0f})"

        # Priority 4: Buy cheap commodities if we have credits and cargo space
        if credits > 1000 and not inventory:
            market = self.get_market()
            if market.get('success'):
                market_data = market.get('market', {})
                commodities = market_data.get('commodities', {})
                # Find cheapest commodity
                best = None
                best_price = float('inf')
                for name, info in commodities.items():
                    price = info.get('buy_price', float('inf'))
                    if price < best_price and price > 0:
                        best = name
                        best_price = price
                if best and best_price < credits * 0.5:
                    qty = min(int(credits * 0.4 / best_price), 20)
                    if qty > 0:
                        # Remember price for this sector
                        self.known_prices[current_sector] = {best: best_price}
                        result = self.buy(best, qty)
                        return f"Bought {qty}x {best} at {best_price}/ea in Sector {current_sector}"

        # Priority 5: Explore - prefer unvisited sectors
        if connections:
            unvisited = [s for s in connections if s not in self.visited_sectors]
            if unvisited:
                target = random.choice(unvisited)
            else:
                target = random.choice(connections)
            result = self.move_to(target)
            prefix = "Explored" if target not in self.visited_sectors else "Moved to"
            return f"{prefix} Sector {target}"

        return "No action available"

    def _play_turn_ai(self, state):
        """Use AI (Grok/OpenAI/compatible) to decide the next action."""
        player = state.get('player', {})
        sector = state.get('sector_info', {})

        # Get market data
        market = self.get_market()
        market_info = market.get('market', {}) if market.get('success') else {}

        prompt = f"""You are an AI agent playing StarTrader Arena, a space trading game.

Current state:
- Sector: {player.get('current_sector')} ({sector.get('type', 'Unknown')})
- Credits: {player.get('credits', 0):.0f}
- Hull: {player.get('hull', 100):.0f}/100
- Fuel: {player.get('fuel', 100)}/{player.get('max_fuel', 100)}
- Inventory: {json.dumps(player.get('inventory', {}))}
- Connected sectors: {sector.get('connections', [])}
- Can refuel here: {sector.get('can_refuel', False)}
- Visited sectors: {sorted(self.visited_sectors)}

Market in this sector:
{json.dumps(market_info.get('commodities', {}), indent=2) if market_info else 'No market data'}

Choose ONE action. Respond with ONLY a JSON object:
{{"action": "move", "sector_id": <int>}}
{{"action": "buy", "commodity": "<name>", "quantity": <int>}}
{{"action": "sell", "commodity": "<name>", "quantity": <int>}}
{{"action": "refuel"}}

Strategy tips:
- Buy low, sell high in different sectors
- Keep hull above 40 (repair when possible)
- Explore new sectors for better prices
- Credits + kills + exploration = your score"""

        try:
            response = self.ai_client.chat.completions.create(
                model=self.ai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            action_text = response.choices[0].message.content.strip()
            # Parse JSON from response
            # Handle cases where AI wraps in ```json
            if '```' in action_text:
                action_text = action_text.split('```')[1]
                if action_text.startswith('json'):
                    action_text = action_text[4:]

            action = json.loads(action_text.strip())

            if action['action'] == 'move':
                result = self.move_to(action['sector_id'])
                self.visited_sectors.add(action['sector_id'])
                return f"AI: Move to Sector {action['sector_id']}"
            elif action['action'] == 'buy':
                result = self.buy(action['commodity'], action['quantity'])
                return f"AI: Buy {action['quantity']}x {action['commodity']}"
            elif action['action'] == 'sell':
                result = self.sell(action['commodity'], action['quantity'])
                return f"AI: Sell {action['quantity']}x {action['commodity']}"
            elif action['action'] == 'refuel':
                result = self.refuel()
                return f"AI: Refuel"
            else:
                return f"AI: Unknown action {action}"

        except Exception as e:
            print(f"    [AI error: {e}] Falling back to heuristic")
            return self._play_turn_heuristic(state)

    # ================================================================
    # Main Loop
    # ================================================================

    def play_match(self):
        """Play through a match until it ends."""
        print("\n[*] Starting match play loop...")
        turn = 0
        consecutive_errors = 0

        while True:
            # Get game state
            state = self.get_state()
            if not state:
                consecutive_errors += 1
                if consecutive_errors > 5:
                    print("[!] Too many consecutive errors. Match may have ended.")
                    break
                time.sleep(3)
                continue

            consecutive_errors = 0
            player = state.get('player', {})
            hull = player.get('hull', 0)

            # Check if we're dead
            if hull <= 0:
                print(f"\n[X] Destroyed after {turn} turns! Final credits: {player.get('credits', 0):.0f}")
                break

            # Check if match is still running
            match_state = self.session.get(f"{self.server}/api/arena/match/state").json()
            if not match_state.get('is_running'):
                print(f"\n[*] Match ended after {turn} turns.")
                # Print final standings
                agents = match_state.get('agents', {})
                ranked = sorted(agents.items(), key=lambda x: x[1].get('score', 0), reverse=True)
                print("\n  Final Standings:")
                for i, (aid, a) in enumerate(ranked):
                    marker = " <-- YOU" if aid == self.agent_id else ""
                    print(f"    {i+1}. {a['name']}: {a.get('score', 0):.0f} pts "
                          f"({a.get('kills', 0)}K/{a.get('deaths', 0)}D, "
                          f"{a.get('credits', 0):.0f}cr){marker}")
                break

            # Play a turn
            action = self.play_turn(state)
            turn += 1
            sector = player.get('current_sector', '?')
            print(f"  Turn {turn:3d} | Sector {sector:3} | "
                  f"Hull {hull:5.0f} | Credits {player.get('credits', 0):8.0f} | {action}")

            # Wait for game speed (match the coordinator's turn timing)
            time.sleep(3)

    def run(self):
        """Full lifecycle: register -> wait -> join -> play."""
        print(f"=" * 60)
        print(f"  StarTrader Arena Agent")
        print(f"  Server: {self.server}")
        print(f"  AI Mode: {'Enabled' if self.use_ai else 'Heuristic only'}")
        print(f"=" * 60)

        # Check if already registered
        if not self.check_registration():
            if not self.register():
                print("[!] Could not register. Exiting.")
                return

        print(f"\n[*] Agent ready: {self.agent_name} ({self.agent_id})")
        print(f"[*] Profile: {self.server}/arena-profile.html?id={self.agent_id}")

        # Wait for match assignment
        if not self.wait_for_match(timeout=600):
            print("[!] No match assigned. Try starting one:")
            print(f'    curl -X POST {self.server}/api/arena/match/start \\')
            print(f'      -H "Content-Type: application/json" \\')
            print(f'      -d \'{{"real_agents": ["{self.agent_id}"]}}\'')
            return

        # Join match (activate game session)
        if not self.join_match():
            print("[!] Could not join match. Exiting.")
            return

        # Play!
        self.play_match()
        print("\n[*] Done! Check your profile:")
        print(f"    {self.server}/arena-profile.html?id={self.agent_id}")


def main():
    parser = argparse.ArgumentParser(description='StarTrader Arena Agent')
    parser.add_argument('--server', default='https://tinycorp.ai',
                        help='Arena server URL (default: https://tinycorp.ai)')
    parser.add_argument('--name', default=None,
                        help='Agent name (default: random)')
    parser.add_argument('--ai', action='store_true',
                        help='Enable AI-powered decisions (requires AI_API_KEY env var)')
    args = parser.parse_args()

    agent = StarTraderAgent(
        server_url=args.server,
        agent_name=args.name,
        use_ai=args.ai
    )
    agent.run()


if __name__ == '__main__':
    main()
